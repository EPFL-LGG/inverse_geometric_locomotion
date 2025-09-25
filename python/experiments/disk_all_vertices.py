"""This script can be executed to explore the motion of a disk under diverse regularization energies.

In a terminal, with the conda environment turned on, run the following command line:

python disk_all_vertices.py --trial_number=8
"""

import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
path_to_cubic_splines = os.path.join(split[0], "inverse_geometric_locomotion/ext/torchcubicspline/")
path_to_notifications = os.path.join(split[0], "inverse_geometric_locomotion/notebooks/notifications/")
path_to_output = os.path.join(split[0], "inverse_geometric_locomotion/output/")
path_to_output_disk = os.path.join(path_to_output, "disk_all_vertices_01_timings/")

if not os.path.exists(path_to_output_disk):
    os.makedirs(path_to_output_disk)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import igl
import numpy as np
from scipy.optimize import minimize, Bounds
import time
import torch

from disk_all_vertices_settings import return_disk_all_vertices_experiment_settings
from geometry_io import export_meshes_to_json
from geometry_generation import generate_disk_mesh
from objectives import compare_last_registration, grad_compare_last_registration
from objectives import surface_area_preservation, grad_surface_area_preservation
from objectives import bending_small_deformation, grad_bending_small_deformation
from objectives import edge_preservation_quad, grad_edge_preservation_quad
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from surface_energy import vmap_compute_mesh_vertex_normals, compute_mesh_vertex_normals, compute_mesh_surface_area, vmap_compute_voronoi_areas, get_internal_edges_and_flaps, compute_mesh_face_areas, compute_mesh_face_normals, compute_mesh_diahedral_angles
from surface_shapes import deformations_generation_cubic_splines
from utils import smooth_hat_function_vary_midpoint, print_quaternion

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", None, "The trial number for the experiment.", lower_bound=0, upper_bound=4, required=True)

def obj_and_grad(
    params, n_ts, n_cp, masses, a_weights, b_weights, 
    faces_mesh, edges_mesh, surf_ref, edge_lengths_ref,
    force_0, torque_0, close_gait, gt,
    vertices_undeformed, vertices_ref_rot, defos_init, free_vertex_dofs,
    i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, 
    weights_surface, weights_edges, delta_i_edges,
    w_fit_scaled, w_area_scaled, w_edges_scaled, w_bend_scaled,
    fun_anisotropy_dir, fun_obj_grad_g,
):
    params_torch = torch.tensor(params)
    params_torch.requires_grad = True

    cps_tmp = params_torch.reshape(n_cp, -1)
    n_free_dofs = cps_tmp.shape[1]
    pos_ = deformations_generation_cubic_splines(
        cps_tmp, torch.tensor(vertices_undeformed), torch.tensor(vertices_ref_rot), defos_init, free_vertex_dofs, n_ts, close_motion=close_gait,
    )
    pos_np_ = pos_.detach().numpy()
    normals_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()

    options = {"maxfev": 2000}
    pos, normals, g = multiple_steps_forward(
        pos_np_, normals_np_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )
    
    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, gt, faces_mesh, edges_mesh, surf_ref, edge_lengths_ref, i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, weights_surface, weights_edges, delta_i_edges, w_fit_scaled, w_area_scaled, w_edges_scaled, w_bend_scaled)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_ts, -1)
    grad_shape_obj = multiple_steps_backward_pos_(
        pos_, pos, normals, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir,
    )
    
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_ts, -1, 3)))
    grad_params_obj = params_torch.grad.numpy()
    grad_params_obj[:n_free_dofs] = 0.0 # First shape is fixed
    
    return obj, grad_params_obj

def fun_obj_grad_g(gs, pos_, gt, faces_mesh, edges_mesh, surf_ref, edge_lengths_ref, i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, weights_surface, weights_edges, delta_i_edges, w_fit_scaled, w_area_scaled, w_edges_scaled, w_bend_scaled):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        gt: torch.tensor of shape (7,)
        faces_mesh: torch.tensor of shape (n_faces, 3)
        edges_mesh: torch.tensor of shape (n_edges, 2)
        surf_ref: torch.tensor representing a scalar
        edge_lengths_ref: torch.tensor of shape (n_edges,)
        i_edge_flaps: torch.tensor of shape (n_i_edges, 2), the connectivity of internal edges to faces
        ref_diahedral_angles: torch.tensor of shape (n_i_edges,), the reference dihedral angles for internal edges
        ref_i_edge_lengths_sq: torch.tensor of shape (n_i_edges,), the reference squared lengths for internal edges
        ref_i_edge_areas: torch.tensor of shape (n_i_edges,), the reference areas for internal edges
        weights_surface: (n_faces,) tensor representing the weights to apply for each face
        weights_edges: (n_edges,) tensor representing the weights to apply for each edge
        delta_i_edges: (n_i_edges,) tensor representing the weights to apply for each internal edge (will be raised to the power 3 in the bending energy: thickness)
        w_*_scaled: float, the weight of the corresponding term
        
    Returns:
        obj: torch.tensor representing a scalar
        grad_g: torch.tensor of shape (n_ts, 7)
        grad_pos_: torch.tensor of shape (n_ts, n_points, 3)
    '''
    obj_fit =  compare_last_registration(gs, gt)
    grad_fit_g = grad_compare_last_registration(gs, gt)
    grad_fit_pos_ = np.zeros_like(pos_)

    if w_area_scaled == 0.0:
        obj_area, grad_area_g, grad_area_pos_ = 0.0, 0.0, 0.0
    else:
        obj_area = surface_area_preservation(pos_, faces_mesh, surf_ref, weights=weights_surface)
        grad_area_g, grad_area_pos_ = grad_surface_area_preservation(pos_, torch.tensor(faces_mesh), surf_ref, weights=weights_surface)

    if w_edges_scaled == 0.0:
        obj_edges, grad_edges_g, grad_edges_pos_ = 0.0, 0.0, 0.0
    else:
        obj_edges = edge_preservation_quad(pos_, edges_mesh, edge_lengths_ref, weights=weights_edges)
        grad_edges_g, grad_edges_pos_ = grad_edge_preservation_quad(pos_, torch.tensor(edges_mesh), edge_lengths_ref, weights=weights_edges)

    if w_bend_scaled == 0.0:
        obj_bend, grad_bend_g, grad_bend_pos_ = 0.0, 0.0, 0.0
    else:
        obj_bend = bending_small_deformation(pos_, faces_mesh, i_edge_flaps, delta_i_edges, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas)
        grad_bend_g, grad_bend_pos_ = grad_bending_small_deformation(pos_, faces_mesh, i_edge_flaps, delta_i_edges, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas)

    obj = w_fit_scaled * obj_fit + w_area_scaled * obj_area + w_edges_scaled * obj_edges + w_bend_scaled * obj_bend
    grad_g = w_fit_scaled * grad_fit_g + w_area_scaled * grad_area_g + w_edges_scaled * grad_edges_g + w_bend_scaled * grad_bend_g
    grad_pos_ = w_fit_scaled * grad_fit_pos_ + w_area_scaled * grad_area_pos_ + w_edges_scaled * grad_edges_pos_ + w_bend_scaled * grad_bend_pos_

    return obj, grad_g, grad_pos_

def main(_):
    
    ###########################################################################
    ## GENERATE THE GEOMETRY
    ###########################################################################

    radius = 1.0
    mesh_size = 0.1
    num_sections = 4
    vertices, faces = generate_disk_mesh(radius, mesh_size, num_sections)
    disk_scaling = np.diag([1.0, 1.0, 1.0])
    disk_scaling[2, 2] = 1.0 / (disk_scaling[0, 0] * disk_scaling[1, 1])
    vertices = vertices @ disk_scaling.T
    torch.manual_seed(0)
    vertices_perturbed = vertices + 1.0e-5 * torch.randn_like(torch.tensor(vertices)).numpy()
    edges_mesh = igl.edges(faces)
    initial_edge_lengths = np.linalg.norm(vertices[edges_mesh[:, 0]] - vertices[edges_mesh[:, 1]], axis=-1)
    aabb = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    aabb_diag = np.linalg.norm(aabb[1] - aabb[0])
    surface_area_aabb = np.prod((aabb[1] - aabb[0])[:2])
    surface_area_disk = compute_mesh_surface_area(torch.tensor(vertices), faces)
    i_edges, i_edge_flaps, i_edge_flap_corners, idx_i_edge, _, edge_map = get_internal_edges_and_flaps(faces)
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    setting_dict = return_disk_all_vertices_experiment_settings(FLAGS.trial_number)
    n_ts, n_cp = setting_dict['n_ts'], setting_dict['n_cp']
    rho, eps, close_gait = setting_dict['rho'], setting_dict['eps'], setting_dict['close_gait']
    max_to_min_weight_ratio = setting_dict['max_to_min_weight_ratio']
    init_perturb_magnitude = setting_dict['init_perturb_magnitude']
    w_fit, w_area, w_edges, w_bend = setting_dict['w_fit'], setting_dict['w_area'], setting_dict['w_edges'], setting_dict['w_bend']
    maxiter = setting_dict['maxiter']
    gt = np.array(setting_dict['gt'])
    
    ###########################################################################
    ## DEFINE WEIGHTS
    ###########################################################################

    min_weight = 2.0 / (1.0 + max_to_min_weight_ratio)
    max_weight = max_to_min_weight_ratio * min_weight

    dist_to_medial_axis = torch.abs(torch.tensor(vertices)[:, 1])
    max_dist_to_medial_axis = torch.max(dist_to_medial_axis)
    scaled_dist_to_medial_axis = dist_to_medial_axis / max_dist_to_medial_axis
    weight_per_vertex = min_weight + (max_weight - min_weight) * smooth_hat_function_vary_midpoint(scaled_dist_to_medial_axis, 0.25)
    weights_surface = torch.mean(weight_per_vertex[faces], dim=1)
    weights_edges = torch.mean(weight_per_vertex[edges_mesh], dim=1)
    weights_i_edges = torch.mean(weight_per_vertex[i_edges], dim=1)
    delta_i_edges = weights_i_edges ** (1.0 / 3.0)

    ref_face_areas = compute_mesh_face_areas(torch.tensor(vertices), torch.tensor(faces))
    ref_i_edge_areas = torch.sum(ref_face_areas[i_edge_flaps], dim=1) / 3.0
    ref_edge_areas = torch.zeros(size=(edges_mesh.shape[0],), dtype=TORCH_DTYPE)
    ref_edge_areas.index_add_(0, torch.tensor(edge_map), torch.repeat_interleave(ref_face_areas / 3.0, repeats=3, dim=0))
    ref_face_normals = compute_mesh_face_normals(torch.tensor(vertices), torch.tensor(faces))
    ref_diahedral_angles = compute_mesh_diahedral_angles(ref_face_normals, i_edge_flaps)
    ref_i_edge_lengths_sq = torch.sum((torch.tensor(vertices)[i_edges[:, 1]] - torch.tensor(vertices)[i_edges[:, 0]])**2, dim=1)

    # Simplify the bending energy
    ref_i_edge_areas = 1.0
    ref_i_edge_lengths_sq = 1.0
    
    ###########################################################################
    ## SIMULATION PARAMETERS
    ###########################################################################
    
    n_points = vertices.shape[0]

    diss_weight = torch.ones(size=(n_ts, n_points))
    a_weights = eps * diss_weight
    b_weights = (1.0 - eps) * diss_weight

    force_0 = np.zeros(shape=(3,))
    torque_0 = np.zeros(shape=(3,))

    def fun_anisotropy_dir(pos_):
        '''
        Args:
            pos_: torch.tensor of shape (n_ts, n_points, 3) or (n_points, 3)
            
        Returns:
            normals_: torch.tensor of shape (n_ts, n_points, 3) or (n_points, 3)
        '''
        if pos_.dim() == 2:
            return compute_mesh_vertex_normals(pos_, torch.tensor(faces))
        else:
            return vmap_compute_mesh_vertex_normals(pos_, torch.tensor(faces))
        
    ###########################################################################
    ## MOTION FROM NOTHING
    ###########################################################################
    
    defos_init = torch.zeros(size=(n_ts, n_points, 3))
    fixed_vertex_dofs = []
    free_vertex_dofs = np.setdiff1d(list(range(3*n_points)), fixed_vertex_dofs).tolist()

    n_free_dofs = len(free_vertex_dofs)
    cps = torch.zeros(n_cp, n_free_dofs)
    torch.manual_seed(0)
    cps += init_perturb_magnitude * torch.randn(n_cp, n_free_dofs)
    cps[0] = 0.0

    pos_ = deformations_generation_cubic_splines(
        cps, torch.tensor(vertices), torch.tensor(vertices_perturbed), defos_init, free_vertex_dofs, n_ts, close_motion=close_gait,
    )

    normals_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()
    voronoi_areas = vmap_compute_voronoi_areas(pos_, torch.tensor(faces))
    masses = rho * np.tile(voronoi_areas[0], reps=(n_ts, 1)) # use the first time step voronoi areas as masses
    
    g0 = np.zeros(shape=(7,))
    g0[3] = 1.0

    pos, tangents, g = multiple_steps_forward(
        pos_.numpy(), normals_, masses, a_weights.numpy(), b_weights.numpy(), force_0, torque_0, g0=g0,
    )

    save_path = os.path.join(path_to_output_disk, "disk_vertices_init_{:02d}.json".format(FLAGS.trial_number))
    export_meshes_to_json(
        pos_, g, pos, force_0, torque_0, edges_mesh, faces, save_path,
        target_final_g=gt,
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    w_fit_scaled = w_fit / aabb_diag ** 2
    w_area_scaled = w_area / (surface_area_disk.item() ** 2)
    w_edges_scaled = w_edges / (np.mean(initial_edge_lengths) ** 2)
    w_bend_scaled = w_bend / (1.0)

    obj_and_grad_scipy = lambda x: obj_and_grad(
        x, n_ts, n_cp, masses, a_weights.numpy(), b_weights.numpy(), 
        faces, edges_mesh, surface_area_disk.item(), torch.tensor(initial_edge_lengths),
        force_0, torque_0, close_gait, gt,
        vertices, vertices_perturbed, defos_init, free_vertex_dofs,
        i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, 
        weights_surface, weights_edges, delta_i_edges,
        w_fit_scaled, w_area_scaled, w_edges_scaled, w_bend_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    
    params_0 = cps.reshape(-1,).numpy()
    lb = -3.0 * np.ones(shape=(n_cp*n_free_dofs,))
    ub = 3.0 * np.ones(shape=(n_cp*n_free_dofs,))

    bnds = Bounds(lb, ub)

    start_time = time.time()
    results_opt = minimize(
        obj_and_grad_scipy, params_0, bounds=bnds, jac=True, method='L-BFGS-B', 
        options={'ftol':1.0e-12, 'gtol':1.0e-7, 'disp': True, 'maxiter': maxiter},
    )
    optim_duration = time.time() - start_time
    
    ###########################################################################
    ## RUN THE OPTIMAL FORWARD PASS AND SAVE THE RESULTS
    ###########################################################################
    
    params_torch_opt = torch.tensor(results_opt.x)

    cps_opt = params_torch_opt.reshape(n_cp, -1)

    pos_opt_ = deformations_generation_cubic_splines(
        cps_opt, torch.tensor(vertices), torch.tensor(vertices_perturbed), defos_init, free_vertex_dofs, n_ts, close_motion=close_gait,
    )

    pos_np_opt_ = pos_opt_.detach().numpy()
    normals_np_opt_ = fun_anisotropy_dir(torch.tensor(pos_np_opt_)).numpy()

    options = {"maxfev": 2000}
    pos_opt, normals_opt, g_opt = multiple_steps_forward(
        pos_np_opt_, normals_np_opt_, masses, a_weights.numpy(), b_weights.numpy(), force_0, torque_0, options=options
    )
    
    save_path = os.path.join(path_to_output_disk, "disk_opt_{:02d}.json".format(FLAGS.trial_number))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_area': w_area,
        'w_area_scaled': w_area_scaled,
        'w_edges': w_edges,
        'w_edges_scaled': w_edges_scaled,
    }

    quantities_per_vertex = {
        'dissipation_weights': diss_weight.tolist(),
        'weight_per_vertex': weight_per_vertex.tolist(),
    }

    quantities_per_face = {
        'weights_surface': weights_surface.tolist(), 
    }

    quantities_per_edge = {
        'weights_edges': weights_edges.tolist(),
    }

    export_meshes_to_json(
        pos_np_opt_, g_opt, pos_opt, force_0, torque_0, edges_mesh, faces, save_path,
        weights_optim=weights_optim, 
        quantities_per_vertex=quantities_per_vertex, quantities_per_face=quantities_per_face,
        quantities_per_edge=quantities_per_edge,
        optimization_settings=setting_dict, optimization_duration=optim_duration,
    )
    
    print("Optimization results:")
    print(g_opt[-1, 4:] - gt[4:])
    print_quaternion(torch.tensor(g[-1, :4]))
    print_quaternion(torch.tensor(g_opt[-1, :4]))
    print_quaternion(torch.tensor(gt[:4]))


if __name__ == '__main__':
    app.run(main)