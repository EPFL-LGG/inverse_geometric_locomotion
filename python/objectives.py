import numpy as np
from obstacle_implicits import ImplicitFunction
from physics_quantities_torch import vmap_energy_torch
from surface_energy import vmap_compute_mesh_volume, vmap_compute_membrane_energy, vmap_compute_bending_energy_small_angles, vmap_compute_bending_energy, vmap_compute_mesh_surface_area, vmap_compute_weighted_mesh_surface_area
import torch
from utils import vmap_euc_transform_torch, vmap_euc_transform_T_torch, vmap_qmultiply_torch, vmap_qbar_torch, register_points_torch_2d, qmultiply_torch, qbar_torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

## Can be useful for a generic gradient computation
def generate_grad_fun(fun):
    '''
    Args:
        fun: a function with signature (pos_, g) -> torch.tensor representing a scalar
        
    Returns:
        grad_fun: a function with signature (pos_, g) -> grad_obj_g, grad_obj_pos_ representing the gradients
    '''

    def grad_fun(pos_, g):
        pos_torch_ = torch.tensor(pos_)
        pos_torch_.requires_grad = True
        
        g_torch = torch.tensor(g)
        g_torch.requires_grad = True

        obj = fun(pos_torch_, g_torch)
        obj.backward(torch.ones_like(obj))
        
        if g_torch.grad is None:
            grad_g = torch.zeros_like(g_torch)
        else:
            grad_g = g_torch.grad
        if pos_torch_.grad is None:
            grad_pos_ = torch.zeros_like(pos_torch_)
        else:
            grad_pos_ = pos_torch_.grad
        return grad_g.numpy(), grad_pos_.numpy()
    
    return grad_fun

def displacement(g, target_norm_g, direction=None):
    '''
        Computes the norm of the translational part g[-1,4:] of the last translation and multiplies by -1. 

        Args:
            g: (n_steps, 7) array representing rigid transformations
            target_norm_g: scalar representing the target norm of the translational part of the last translation
            direction: (3,) array representing the direction of the displacement: should be normalized

        Returns:
            scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g

    if direction is not None:
        if type(direction) is np.ndarray:
            direction_torch = torch.tensor(direction)
        else:
            direction_torch = direction
        return torch.abs(target_norm_g ** 2 - torch.sum(g_torch[-1, 4:] * direction_torch) ** 2)
    else:
        return torch.abs(target_norm_g ** 2 - torch.sum(g_torch[-1, 4:] ** 2))

def grad_displacement(g, target_norm_g, direction=None):
    '''
        Gradient of the displacement

        Args:
            Same as above

        Returns:
            (n_steps, 7) array representing the gradient of the displacement
    '''

    g_torch = torch.tensor(g)
    g_torch.requires_grad = True
    #gcp_torch = torch.tensor(gcp)

    obj = displacement(g_torch, target_norm_g, direction=direction)
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy()


def compare_last_translation(g, gt):
    '''Compares the translation to the target translation
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (7,) array representing the target rigid transformation
        
    Returns:
        scalar representing the objective
    '''
    return np.sum((g[-1, 4:] - gt[4:]) ** 2) / 2.0

def grad_compare_last_translation(g, gt):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    grad = np.zeros_like(g)
    grad[-1, 4:] = g[-1, 4:] - gt[4:]
    return grad

def compare_last_orientation(g, gt):
    '''Compares the rotation to the target rotation

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (7,) array representing the target rigid transformation
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    # since the quaternions are normalized, the inverse is the conjugate
    diff_quat = qmultiply_torch(g_torch[-1, :4], qbar_torch(gt_torch[:4]))
    angle = 2.0 * torch.atan2(torch.linalg.norm(diff_quat[:3]), diff_quat[3])

    return 0.5 * torch.minimum(
        angle, torch.tensor(2.0 * np.pi) - angle
    ) ** 2

def grad_compare_last_orientation(g, gt):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True
    gt_torch = torch.tensor(gt)

    obj = compare_last_orientation(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy()

def compare_all_translations(g, gt):
    '''Compares the translation to the target translation
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    return np.sum((g[:, 4:] - gt[:, 4:]) ** 2) / 2.0

def grad_compare_all_translations(g, gt):
    '''Gradient of the objective with respect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    grad = np.zeros_like(g)
    grad[:, 4:] = g[:, 4:] - gt[:, 4:]
    return grad

def compare_all_orientations(g, gt):
    '''Compares the rotations to the target rotations

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    diff_quats = vmap_qmultiply_torch(
        g_torch[:, :4],
        vmap_qbar_torch(gt_torch[:, :4]) # since the quaternions are normalized, the inverse is the conjugate
    )

    angles = 2.0 * torch.atan2(
        torch.linalg.norm(diff_quats[:, :3], dim=1),
        diff_quats[:, 3]
    )

    return torch.sum(torch.minimum(
        angles, torch.tensor(2.0 * np.pi) - angles
    ) ** 2) / 2.0

def grad_compare_all_orientations(g, gt):
    '''Gradient of the objective with respect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    gt_torch = torch.tensor(gt)

    obj = compare_all_orientations(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    
    return g_torch.grad.numpy()

def compare_all_translation_increments(g, gt):
    '''Compares the rotations increments to the target rotations increments

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    trans_increments = g_torch[1:, 4:] - g_torch[:-1, 4:]
    transt_increments = gt_torch[1:, 4:] - gt_torch[:-1, 4:]

    return torch.sum((trans_increments - transt_increments) ** 2) / 2.0

def grad_compare_all_translation_increments(g, gt):
    '''Gradient of the objective with respect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    gt_torch = torch.tensor(gt)

    obj = compare_all_translation_increments(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    
    return g_torch.grad.numpy()

def compare_all_orientation_increments(g, gt):
    '''Compares the rotations increments to the target rotations increments

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    quats_increments = vmap_qmultiply_torch(
        g_torch[1:, :4],
        vmap_qbar_torch(g_torch[:-1, :4]) # detach here is to be discussed
    )

    quatst_increments = vmap_qmultiply_torch(
        gt_torch[1:, :4],
        vmap_qbar_torch(gt_torch[:-1, :4])
    )

    diff_quats = vmap_qmultiply_torch(
        quats_increments,
        vmap_qbar_torch(quatst_increments)
    )

    angles = 2.0 * torch.atan2(
        torch.linalg.norm(diff_quats[:, :3], dim=1),
        diff_quats[:, 3]
    )

    return torch.sum(torch.minimum(
        angles, torch.tensor(2.0 * np.pi) - angles
    ) ** 2) / 2.0

def grad_compare_all_orientation_increments(g, gt):
    '''Gradient of the objective with respect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    gt_torch = torch.tensor(gt)

    obj = compare_all_orientation_increments(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    
    return g_torch.grad.numpy()

def compare_all_registrations(g, gt, norm_trans_weight=1.0, norm_rot_weight=1.0):
    '''Compares the registration to the target registration

    Note: use them separately since the two terms would need different normalizations
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        norm_trans_weight: scalar weight for the translation part
        norm_rot_weight: scalar weight for the rotation part

    Returns:
        scalar representing the objective
    '''
    return norm_trans_weight * compare_all_translations(g, gt) + norm_rot_weight * compare_all_orientations(g, gt).detach().item()

def grad_compare_all_registrations(g, gt, norm_trans_weight=1.0, norm_rot_weight=1.0):
    '''Gradient of the objective with respect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    return norm_trans_weight * grad_compare_all_translations(g, gt) + norm_rot_weight * grad_compare_all_orientations(g, gt)

def compare_last_registration(g, gt, norm_trans_weight=1.0, norm_rot_weight=1.0):
    '''Compares the registration to the target registration
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (7,) array representing the target rigid transformation
        norm_trans_weight: scalar weight for the translation part
        norm_rot_weight: scalar weight for the rotation part
        
    Returns:
        scalar representing the objective
    '''
    return norm_trans_weight * compare_last_translation(g, gt) + norm_rot_weight * compare_last_orientation(g, gt)

def grad_compare_last_registration(g, gt, norm_trans_weight=1.0, norm_rot_weight=1.0):
    '''Gradient of the objective with respect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    return norm_trans_weight * grad_compare_last_translation(g, gt) + norm_rot_weight * grad_compare_last_orientation(g, gt)

def compare_to_target_gait(pos_, pos_target_):
    '''Compute the deviation to the target gait
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        pos_target_: (n_steps, n_points, 3) array of the target positions of the system
        
    Returns:
        scalar representing the objective
    '''
    return np.sum((pos_ - pos_target_) ** 2) / 2.0

def grad_compare_to_target_gait(pos_, pos_target_):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    grad_obj_pos_ = pos_ - pos_target_
    return np.zeros(shape=(pos_.shape[0], 7)), grad_obj_pos_

def com_displacement_sq(g):
    '''Compute the sum of the squared displacement of the center of mass compared to the initial center of mass position
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    return np.mean(np.sum((g[:, 4:] - g[0, 4:]) ** 2, axis=1)) / 2.0

def grad_com_displacement_sq(g):
    '''Gradient of the objective with repect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    grad = np.zeros_like(g)
    grad[:, 4:] = (g[:, 4:] - g[0, 4:]) / g.shape[0]
    return grad

def compute_com_bending(pos_, g):
    '''Computes the bending of the center of mass trajectory
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) array representing rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(pos_) is np.ndarray:
        pos_torch_ = torch.tensor(pos_)
    else:
        pos_torch_ = pos_
    trans_edges = g_torch[1:, 4:] - g_torch[:-1, 4:]
    turning_angles = torch.atan2(torch.linalg.norm(torch.cross(trans_edges[1:], trans_edges[:-1], dim=1), dim=1), torch.sum(trans_edges[1:] * trans_edges[:-1], dim=1))
    return torch.mean(turning_angles ** 2) / 2.0 + 0.0 * pos_torch_[0, 0, 0] # To make sure the gradient is not None

grad_compute_com_bending = generate_grad_fun(compute_com_bending)


def pass_checkpoints(g, gcp):
    '''Makes sure the rigid transformations pass through the checkpoints
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gcp: (n_cp, 7) array representing the checkpoints
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gcp) is np.ndarray:
        gcp_torch = torch.tensor(gcp)
    else:
        gcp_torch = gcp
    
    dists_to_cp = torch.sum((g_torch.unsqueeze(1)[..., 4:] - gcp_torch.unsqueeze(0)[..., 4:]) ** 2, dim=-1) # (n_steps, n_cp)
    min_dists_to_cp = torch.mean(torch.min(dists_to_cp, dim=0)[0])
    # min_dists_to_cp = torch.mean(torch.min(dists_to_cp, dim=0)[0]) + torch.mean(torch.min(dists_to_cp, dim=1)[0]) # (n_cp,)
    return min_dists_to_cp / 2.0

def grad_pass_checkpoints(g, gcp):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True
    gcp_torch = torch.tensor(gcp)

    obj = pass_checkpoints(g_torch, gcp_torch)
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy()

def penalize_upper_bound(params, upper_bound):
    '''Penalizes the parameters that are above the upper bound quadratically'''
    return 0.5 * torch.sum(torch.relu(params - upper_bound) ** 2)

def energy_path(pos_, g, masses, a_weight, b_weight, fun_anisotropy_dir):
    '''Compute the energy of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) array representing the rigid transformation
        masses: (n_steps, n_points, 1) array of the masses of the system
        a_weight: scalar or (n_points, 1) array representing the a parameters for anisotropy of local dissipations metrics
        b_weight: scalar or (n_points, 1) array representing the b parameters for anisotropy of local dissipations metrics
        fun_anisotropy_dir: callable that computes the anisotropy direction of the system
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
    if not isinstance(masses, torch.Tensor):
        masses = torch.tensor(masses)
    if not isinstance(a_weight, torch.Tensor):
        a_weight = torch.tensor(a_weight)
    if not isinstance(b_weight, torch.Tensor):
        b_weight = torch.tensor(b_weight)

    tangents_ = fun_anisotropy_dir(pos_)
    pos = vmap_euc_transform_torch(g, pos_)
    tangents = vmap_euc_transform_T_torch(g, tangents_)

    energies = vmap_energy_torch(
        pos[:-1], pos[1:], tangents[:-1], tangents[1:], 
        masses[:-1], masses[1:], a_weight[:-1], a_weight[1:], 
        b_weight[:-1], b_weight[1:],
    )

    return torch.sum(energies)

def grad_energy_path(pos_, g, masses, a_weight, b_weight, fun_anisotropy_dir):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_, requires_grad=True)
    g_torch = torch.tensor(g, requires_grad=True)
    masses_torch = torch.tensor(masses)
    a_weight_torch = torch.tensor(a_weight)
    b_weight_torch = torch.tensor(b_weight)

    obj = energy_path(
        pos_torch_, g_torch, 
        masses_torch, a_weight_torch, b_weight_torch, fun_anisotropy_dir
    )
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy(), pos_torch_.grad.numpy()

def discrete_g_speed_smoothing(pos_, g):
    '''Compute the variance of mean speed along the trajectory
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
    speed = torch.sqrt(torch.sum((g[1:, 4:] - g[:-1, 4:]) ** 2, dim=-1) + 1.0e-7)
    return torch.var(speed) + 0.0 * pos_[0, 0, 0] # To make sure the gradient is not None

grad_discrete_g_speed_smoothing = generate_grad_fun(discrete_g_speed_smoothing)

def volume_preservation(pos_, faces, initial_volume):
    '''Compute the volume preservation of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        faces: (n_faces, 3) tensor of the faces of the mesh
        initial_volume: scalar representing the initial volume of the system
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    volume_preservation = 0.5 * torch.mean((vmap_compute_mesh_volume(pos_, faces) - initial_volume) ** 2)
    return volume_preservation

def grad_volume_preservation(pos_, faces, initial_volume):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = volume_preservation(pos_torch_, faces, initial_volume)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def surface_area_preservation(pos_, faces, initial_surface_area, weights=None):
    '''Compute the volume preservation of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        faces: (n_faces, 3) tensor of the faces of the mesh
        initial_surface_area: scalar representing the initial surface area of the system
        weights: (n_faces,) tensor representing the weights to apply for each face
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    if weights is not None:
        surface_area_preservation = 0.5 * torch.mean((vmap_compute_weighted_mesh_surface_area(pos_, faces, weights) - initial_surface_area) ** 2)
    else:
        surface_area_preservation = 0.5 * torch.mean((vmap_compute_mesh_surface_area(pos_, faces) - initial_surface_area) ** 2)
    return surface_area_preservation

def grad_surface_area_preservation(pos_, faces, initial_surface_area, weights=None):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = surface_area_preservation(pos_torch_, faces, initial_surface_area, weights=weights)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def edge_preservation(pos_, edges, initial_edge_lengths, weights=None):
    '''Compute the volume preservation of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        edges: (n_edges, 2) tensor of the edges of the mesh
        initial_edge_lengths: (n_edges,) tensor representing the initial edge lengths of the system
        weights: (n_edges,) tensor representing the weights to apply for each edge
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    new_edge_lengths_squared = torch.sum((pos_[:, edges[:, 0]] - pos_[:, edges[:, 1]]) ** 2, dim=-1)
    if weights is not None:
        edge_preservation = 0.25 * torch.mean(weights.reshape(1, -1) * (new_edge_lengths_squared - initial_edge_lengths ** 2) ** 2)
    else:
        edge_preservation = 0.25 * torch.mean((new_edge_lengths_squared - initial_edge_lengths ** 2) ** 2)
    return edge_preservation

def grad_edge_preservation(pos_, edges, initial_edge_lengths, weights=None):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = edge_preservation(pos_torch_, edges, initial_edge_lengths, weights=weights)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def repulsive_curve(pos_, alpha, beta, n_edge_disc=0):
    '''Compute the repulsion of the curve with itself

    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        alpha: exponent of the repulsion kernel for the distance between points along the tangent direction
        beta: exponent of the repulsion kernel for the distance between points
        n_edge_disc: number of additional discretization points along the edges (the original paper assumes n_edge_disc=0)

    Returns:
        scalar representing the objective
    '''
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_, dtype=TORCH_DTYPE)
    n_steps = pos_.shape[0]
    n_pts = pos_.shape[1]
    edges = torch.stack([torch.arange(n_pts - 1), torch.arange(1, n_pts)], dim=-1)
    n_edges = edges.shape[0]
    alphas = torch.linspace(0.0, 1.0, n_edge_disc+2).reshape(1, 1, -1, 1) # shape (n_edge_disc+2,)
    edge_disc = pos_[:, edges[:, 0]].unsqueeze(2) * (1.0 - alphas) + pos_[:, edges[:, 1]].unsqueeze(2) * alphas # shape (n_steps, n_edges, n_edge_disc, 3)
    edge_tangents = (pos_[:, edges[:, 1]] - pos_[:, edges[:, 0]]) # shape (n_steps, n_edges, 3)
    edge_lengths = torch.sqrt(1.0e-12 + torch.sum(edge_tangents ** 2, dim=-1)) # shape (n_steps, n_edges)
    edge_lengths_int_weights = edge_lengths.reshape(n_steps, n_edges, 1) * edge_lengths.reshape(n_steps, 1, n_edges) # shape (n_steps, n_edges, n_edges)
    edge_tangents = edge_tangents / edge_lengths.unsqueeze(-1) # shape (n_steps, n_edges, 3)
    edge_disc_pd_disp = (edge_disc.reshape(pos_.shape[0], n_edges, 1, n_edge_disc+2, 1, 3) - edge_disc.reshape(pos_.shape[0], 1, n_edges, 1, n_edge_disc+2, 3)) # shape (n_steps, n_edges, n_edges, n_edge_disc+2, n_edge_disc+2, 3)
    edge_disc_pd = torch.sqrt(1.0e-12 + torch.sum(edge_disc_pd_disp ** 2, dim=-1)) # shape (n_steps, n_edges, n_edges, n_edge_disc+2, n_edge_disc+2)
    edge_disc_normal_proj = torch.sqrt(1.0e-12 + torch.sum(edge_disc_pd_disp - torch.sum(edge_disc_pd_disp * edge_tangents.reshape(n_steps, n_edges, 1, 1, 1, 3), dim=-1, keepdim=True) * edge_tangents.reshape(n_steps, n_edges, 1, 1, 1, 3), dim=-1) ** 2) # shape (n_steps, n_edges, n_edges, n_edge_disc+2, n_edge_disc+2)

    idx_upper_edges = torch.triu_indices(n_edges, n_edges, offset=2) # do not pick neighboring edges, as in the paper
    int_weights = edge_lengths_int_weights[:, idx_upper_edges[0], idx_upper_edges[1]] # shape (n_steps, (n_edges-1) * (n_edges-2) / 2)
    numerator = edge_disc_normal_proj[:, idx_upper_edges[0], idx_upper_edges[1], :, :] ** alpha # shape (n_steps, (n_edges-1) * (n_edges-2) / 2, n_edge_disc+2, n_edge_disc+2)
    denominator = edge_disc_pd[:, idx_upper_edges[0], idx_upper_edges[1], :, :] ** beta # shape (n_steps, (n_edges-1) * (n_edges-2) / 2, n_edge_disc+2, n_edge_disc+2)
    idx_upper_edge_disc = torch.triu_indices(n_edge_disc+2, n_edge_disc+2, offset=0)
    repulsion = torch.mean(numerator[:, :, idx_upper_edge_disc[0], idx_upper_edge_disc[1]] / (1.0e-12 + denominator[:, :, idx_upper_edge_disc[0], idx_upper_edge_disc[1]]), dim=-1) # shape (n_steps, (n_edges-1) * (n_edges-2) / 2)
    return torch.mean(int_weights * repulsion)

def grad_repulsive_curve(pos_, alpha, beta, n_edge_disc=0):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = repulsive_curve(pos_torch_, alpha, beta, n_edge_disc=n_edge_disc)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def edge_centers_repulsion(pos_, edges, length_threshold, n_edge_disc=1):
    '''Compute the edge repulsion of the path, introducing some points along the edges.
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        edges: (n_edges, 2) tensor of the edges of the graph
        length_threshold: scalar representing the length threshold
        
    Returns:
        scalar representing the objective
    '''
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_, dtype=TORCH_DTYPE)
        
    n_pts = pos_.shape[1]
    idx_upper_pts = torch.triu_indices(n_pts, n_pts, offset=1)
    pairwise_distances_pts = torch.sqrt(1.0e-12 + torch.sum((pos_[:, idx_upper_pts[0]] - pos_[:, idx_upper_pts[1]]) ** 2, dim=-1)) # shape (n_steps, n_pts * (n_pts-1) / 2)
    energy_pts = 0.5 * torch.mean(torch.relu(- (pairwise_distances_pts - length_threshold) ** 2 * torch.log(pairwise_distances_pts / length_threshold))) # IPC
    
    energy_edges = 0.0
    if n_edge_disc > 0:
        n_edges = edges.shape[0]
        alphas = torch.linspace(0.0, 1.0, n_edge_disc + 2)[1:-1].reshape(1, 1, -1, 1) # shape (n_edge_disc,)
        edge_disc = pos_[:, edges[:, 0]].unsqueeze(2) * (1.0 - alphas) + pos_[:, edges[:, 1]].unsqueeze(2) * alphas # shape (n_steps, n_edges, n_edge_disc, 3)
        edge_disc_pd = torch.sqrt(1.0e-12 + torch.sum((edge_disc.reshape(pos_.shape[0], n_edges, 1, n_edge_disc, 1, 3) - edge_disc.reshape(pos_.shape[0], 1, n_edges, 1, n_edge_disc, 3)) ** 2, dim=-1)) # shape (n_steps, n_edges, n_edges, n_edge_disc, n_edge_disc)
        idx_upper_edge_disc = torch.triu_indices(n_edge_disc, n_edge_disc, offset=0)
        edge_to_edge_min_dist = torch.min(edge_disc_pd[:, :, :, idx_upper_edge_disc[0], idx_upper_edge_disc[1]], dim=-1)[0] # shape (n_steps, n_edges, n_edges)
        idx_upper_edges = torch.triu_indices(n_edges, n_edges, offset=1)
        pairwise_distances_edges = edge_to_edge_min_dist[:, idx_upper_edges[0], idx_upper_edges[1]] # shape (n_steps, n_edges * (n_edges-1) / 2)
        energy_edges = 0.5 * torch.mean(torch.relu(- (pairwise_distances_edges - length_threshold) ** 2 * torch.log(pairwise_distances_edges / length_threshold))) # IPC
    
    return energy_edges + energy_pts
    
def grad_edge_centers_repulsion(pos_, edges, length_threshold, n_edge_disc=1):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = edge_centers_repulsion(pos_torch_, edges, length_threshold, n_edge_disc=n_edge_disc)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def edge_preservation_quad(pos_, edges, initial_edge_lengths, weights=None):
    '''Compute the edge preservation of the path using a quadratic loss
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        edges: (n_edges, 2) tensor of the edges of the mesh
        initial_edge_lengths: (n_edges,) tensor representing the initial edge lengths of the system
        weights: (n_edges,) tensor representing the weights to apply for each edge
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_, dtype=TORCH_DTYPE)
    new_edge_lengths = torch.linalg.norm((pos_[:, edges[:, 0]] - pos_[:, edges[:, 1]]), dim=-1)
    if weights is not None:
        edge_preservation = 0.5 * torch.mean(weights.reshape(1, -1) * (new_edge_lengths - initial_edge_lengths) ** 2)
    else:
        edge_preservation = 0.5 * torch.mean((new_edge_lengths - initial_edge_lengths) ** 2)
    return edge_preservation

def grad_edge_preservation_quad(pos_, edges, initial_edge_lengths, weights=None):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_, dtype=TORCH_DTYPE)
    pos_torch_.requires_grad = True

    obj = edge_preservation_quad(pos_torch_, edges, initial_edge_lengths, weights=weights)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def bending_small_deformation(pos_, faces, face_adjacency, delta_per_i_edge, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''Compute the volume preservation of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        faces: (n_f, 3) tensor representing the faces of the mesh
        face_adjacency: (n_face_pairs, 2) array of the face adjacency
        delta_per_i_edge: float or (n_face_pairs,) representing the width of the shell
        ref_diahedral_angles: (n_face_pairs,) tensor representing the reference dihedral angles
        ref_edge_lengths_sq: (n_face_pairs,) tensor representing the reference edge squared lengths between adjacent faces
        ref_edge_areas: (n_face_pairs,) tensor representing the reference edge areas between adjacent faces
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    
    obj_bending = torch.mean(vmap_compute_bending_energy_small_angles(pos_, faces, face_adjacency, delta_per_i_edge, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas))
    return obj_bending

def grad_bending_small_deformation(pos_, faces, face_adjacency, delta_per_i_edge, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = bending_small_deformation(pos_torch_, faces, face_adjacency, delta_per_i_edge, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def membrane_bending_small_deformation(pos_, faces, face_adjacency,  delta_per_face, delta_per_i_edge, lambda_, mu, ref_inv_ff, ref_areas, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''Compute the volume preservation of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        faces: (n_f, 3) tensor representing the faces of the mesh
        face_adjacency: (n_face_pairs, 2) array of the face adjacency
        delta_per_face: float or (n_f,) representing the width of the shell
        delta_per_i_edge: float or (n_face_pairs,) representing the width of the shell
        lambda_: first lame parameter
        mu: second lame parameter
        ref_inv_ff: (n_f, 3, 3) tensor representing the inverse of the reference first fundamental form
        ref_areas: (n_f,) tensor representing the reference areas of the faces
        ref_diahedral_angles: (n_face_pairs,) tensor representing the reference dihedral angles
        ref_edge_lengths_sq: (n_face_pairs,) tensor representing the reference edge squared lengths between adjacent faces
        ref_edge_areas: (n_face_pairs,) tensor representing the reference edge areas between adjacent faces
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    
    obj_membrane = torch.mean(vmap_compute_membrane_energy(pos_, faces, delta_per_face, lambda_, mu, ref_inv_ff, ref_areas))
    obj_bending = torch.mean(vmap_compute_bending_energy_small_angles(pos_, faces, face_adjacency, delta_per_i_edge, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas))
    return obj_membrane + obj_bending

def grad_membrane_bending_small_deformation(pos_, faces, face_adjacency, delta_per_face, delta_per_i_edge, lambda_, mu, ref_inv_ff, ref_areas, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = membrane_bending_small_deformation(pos_torch_, faces, face_adjacency,  delta_per_face, delta_per_i_edge, lambda_, mu, ref_inv_ff, ref_areas, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas)
    obj.backward(torch.ones_like(obj))
    return np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def obj_grad_membrane_bending_small_deformation(pos_, faces, face_adjacency, delta_per_face, delta_per_i_edge, lambda_, mu, ref_inv_ff, ref_areas, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True

    obj = membrane_bending_small_deformation(pos_torch_, faces, face_adjacency,  delta_per_face, delta_per_i_edge, lambda_, mu, ref_inv_ff, ref_areas, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas)
    obj.backward(torch.ones_like(obj))
    return obj.detach().item(), np.zeros(shape=(pos_.shape[0], 7)), pos_torch_.grad.numpy()

def avoid_implicit(pos_: torch.Tensor, g: torch.Tensor, implicit: ImplicitFunction):
    '''Quantify the penetration of an object to a given implicit function
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) tensor reprenting the registration per time step
        implicit (instance of an ImplicitFunction): the obstacle to avoid
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
        
    pos = vmap_euc_transform_torch(g, pos_)
    sdfs = implicit.evaluate_implicit_function(pos.reshape(-1, 3))
    obj = torch.mean(torch.maximum(torch.tensor(0.0), - sdfs))
    return obj

def grad_avoid_implicit(pos_: torch.Tensor, g: torch.Tensor, implicit: ImplicitFunction):
    '''Quantify the penetration of an object to a given implicit function
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) tensor reprenting the registration per time step
        implicit (instance of an ImplicitFunction): the obstacle to avoid
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
        
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    obj = avoid_implicit(pos_, g_torch, implicit)
    obj.backward(torch.ones_like(obj))

    # print(pos_torch_.grad.shape)
    # print(g_torch.grad.shape)
    
    if g_torch.grad is None:
        grad_g = torch.zeros_like(g_torch)
    else:
        grad_g = g_torch.grad
    if pos_torch_.grad is None:
        grad_pos_ = torch.zeros_like(pos_torch_)
    else:
        grad_pos_ = pos_torch_.grad
    return grad_g.numpy(), grad_pos_.numpy()

def smooth_avoid_implicit(pos_: torch.Tensor, g: torch.Tensor, implicit: ImplicitFunction, beta: float=1.0):
    '''Compute the avoidance of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) tensor reprenting the registration per time step
        implicit (instance of an ImplicitFunction): the obstacle to avoid
        beta: scalar representing the beta parameter for the softplus function
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
        
    pos = vmap_euc_transform_torch(g, pos_)
    sdfs = implicit.evaluate_implicit_function(pos.reshape(-1, 3))
    obj = torch.mean(torch.nn.functional.softplus(-sdfs, beta=beta))
    return obj

def grad_smooth_avoid_implicit(pos_: torch.Tensor, g: torch.Tensor, implicit: ImplicitFunction, beta: float=1.0):
    '''Compute the avoidance of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) tensor reprenting the registration per time step
        implicit (instance of an ImplicitFunction): the obstacle to avoid
        beta: scalar representing the beta parameter for the softplus function
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
        
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    obj = smooth_avoid_implicit(pos_, g_torch, implicit, beta=beta)
    obj.backward(torch.ones_like(obj))
    
    if g_torch.grad is None:
        grad_g = torch.zeros_like(g_torch)
    else:
        grad_g = g_torch.grad
    if pos_torch_.grad is None:
        grad_pos_ = torch.zeros_like(pos_torch_)
    else:
        grad_pos_ = pos_torch_.grad
    return grad_g.numpy(), grad_pos_.numpy()
