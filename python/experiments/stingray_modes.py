"""This script can be executed to teach a stingray flapping its pectoral fins.

In a terminal, with the conda environment turned on, run the following command line:

python stingray_modes.py --trial_number=0
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
path_to_output_stingray = os.path.join(path_to_output, "stingray_vibrational/")
path_to_data_stingray = os.path.join(split[0], "inverse_geometric_locomotion/data/")

if not os.path.exists(path_to_output_stingray):
    os.makedirs(path_to_output_stingray)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import igl
import json
import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize, Bounds
import time
import torch
from torch.autograd.functional import hessian

from stingray_modes_settings import return_stingray_modes_experiment_settings
from geometry_io import export_meshes_to_json, read_mesh_obj_triangles_quads
from objectives import compare_last_registration, grad_compare_last_registration
from objectives import pass_checkpoints, grad_pass_checkpoints
from objectives import volume_preservation, grad_volume_preservation
from objectives import bending_small_deformation, grad_bending_small_deformation
from objectives import edge_preservation_quad, grad_edge_preservation_quad
from objectives import discrete_g_speed_smoothing, grad_discrete_g_speed_smoothing
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from surface_energy import vmap_compute_mesh_vertex_normals, compute_mesh_vertex_normals, vmap_compute_voronoi_areas, get_internal_edges_and_flaps, compute_mesh_face_areas, compute_mesh_face_normals, compute_mesh_diahedral_angles, compute_mesh_volume
from surface_shapes import elasticity_hessian_modes_generation_cubic_splines
from utils import smooth_hat_function_vary_midpoint, print_quaternion, align_point_cloud

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", 0, "The trial number for the experiment.")
flags.DEFINE_boolean("recompute_modes", False, "Whether to recompute the modes.")

def obj_and_grad(
    params, n_ts, n_cp, masses, a_weights, b_weights, 
    faces_mesh, edges_mesh, volume_ref, edge_lengths_ref,
    force_0, torque_0, close_gait, gt, gcp,
    vertices_undeformed, modes_torch,
    i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, 
    weights_edges, delta_i_edges,
    w_fit_scaled, w_fit_cp_scaled, w_vol_scaled, w_edges_scaled, w_bend_scaled, w_g_speed_scaled,
    fun_anisotropy_dir, fun_obj_grad_g,
):
    params_torch = torch.tensor(params)
    params_torch.requires_grad = True

    cps_tmp = params_torch.reshape(n_cp, -1)
    pos_ = elasticity_hessian_modes_generation_cubic_splines(cps_tmp, modes_torch, vertices_undeformed, n_ts, close_motion=close_gait)
    pos_np_ = pos_.detach().numpy()
    normals_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()

    options = {"maxfev": 2000}
    pos, normals, g = multiple_steps_forward(
        pos_np_, normals_np_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )
    
    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, gt, gcp, faces_mesh, edges_mesh, volume_ref, edge_lengths_ref, i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, weights_edges, delta_i_edges, w_fit_scaled, w_fit_cp_scaled, w_vol_scaled, w_edges_scaled, w_bend_scaled, w_g_speed_scaled)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_ts, -1)
    grad_shape_obj = multiple_steps_backward_pos_(
        pos_, pos, normals, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir,
    )
    
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_ts, -1, 3)))
    grad_params_obj = params_torch.grad.numpy()
    
    return obj, grad_params_obj

def fun_obj_grad_g(gs, pos_, gt, gcp, faces_mesh, edges_mesh, volume_ref, edge_lengths_ref, i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, weights_edges, delta_i_edges, w_fit_scaled, w_fit_cp_scaled, w_vol_scaled, w_edges_scaled, w_bend_scaled, w_g_speed_scaled):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        gt: torch.tensor of shape (7,)
        gcp: torch.tensor of shape (n_cp, 7)
        faces_mesh: torch.tensor of shape (n_faces, 3)
        edges_mesh: torch.tensor of shape (n_edges, 2)
        volume_ref: torch.tensor representing a scalar
        edge_lengths_ref: torch.tensor of shape (n_edges,)
        i_edge_flaps: torch.tensor of shape (n_i_edges, 2), the connectivity of internal edges to faces
        ref_diahedral_angles: torch.tensor of shape (n_i_edges,), the reference dihedral angles for internal edges
        ref_i_edge_lengths_sq: torch.tensor of shape (n_i_edges,), the reference squared lengths for internal edges
        ref_i_edge_areas: torch.tensor of shape (n_i_edges,), the reference areas for internal edges
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

    if w_fit_cp_scaled == 0.0:
        obj_fit_cp, grad_fit_cp_g, grad_fit_cp_pos_ = 0.0, 0.0, 0.0
    else:
        obj_fit_cp = pass_checkpoints(gs, gcp)
        grad_fit_cp_g = grad_pass_checkpoints(gs, gcp)
        grad_fit_cp_pos_ = np.zeros_like(pos_)

    if w_vol_scaled == 0.0:
        obj_vol, grad_vol_g, grad_vol_pos_ = 0.0, 0.0, 0.0
    else:
        obj_vol = volume_preservation(pos_, faces_mesh, volume_ref)
        grad_vol_g, grad_vol_pos_ = grad_volume_preservation(pos_, faces_mesh, volume_ref)

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

    if w_g_speed_scaled == 0.0:
        obj_g_speed, grad_g_speed_g, grad_g_speed_pos_ = 0.0, 0.0, 0.0
    else:
        obj_g_speed = discrete_g_speed_smoothing(pos_, gs)
        grad_g_speed_g, grad_g_speed_pos_ = grad_discrete_g_speed_smoothing(pos_, gs)

    obj = w_fit_scaled * obj_fit + w_fit_cp_scaled * obj_fit_cp + w_vol_scaled * obj_vol + w_edges_scaled * obj_edges + w_bend_scaled * obj_bend + w_g_speed_scaled * obj_g_speed
    grad_g = w_fit_scaled * grad_fit_g + w_fit_cp_scaled * grad_fit_cp_g + w_vol_scaled * grad_vol_g + w_edges_scaled * grad_edges_g + w_bend_scaled * grad_bend_g + w_g_speed_scaled * grad_g_speed_g
    grad_pos_ = w_fit_scaled * grad_fit_pos_ + w_fit_cp_scaled * grad_fit_cp_pos_ + w_vol_scaled * grad_vol_pos_ + w_edges_scaled * grad_edges_pos_ + w_bend_scaled * grad_bend_pos_ + w_g_speed_scaled * grad_g_speed_pos_

    return obj, grad_g, grad_pos_

def main(_):
    
    ###########################################################################
    ## GENERATE THE GEOMETRY
    ###########################################################################

    file_name = os.path.join(path_to_data_stingray, "Stingray_LOD0.obj")
    vertices, f_triangles, f_quads = read_mesh_obj_triangles_quads(file_name)
    vertices = align_point_cloud(torch.tensor(vertices)).numpy()
    # The tail should point to -x
    if abs(np.min(vertices[0])) < abs(np.max(vertices[0])):
        vertices[:, 0] = -vertices[:, 0]
        vertices[:, 1] = -vertices[:, 1]

    aabb = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    aabb_diag = np.linalg.norm(aabb[1] - aabb[0])
    stringray_width = aabb[1, 1] - aabb[0, 1]
    vertices = 2.0 * vertices / stringray_width
    vertices_ref = torch.tensor(vertices)
    torch.manual_seed(0)
    vertices_perturbed = vertices + 1.0e-5 * torch.randn_like(torch.tensor(vertices)).numpy()

    faces = np.concatenate([
        f_triangles, 
        f_quads[:, [0, 1, 2]], 
        f_quads[:, [0, 2, 3]]
    ], axis=0)
    edges_mesh = igl.edges(faces)
    initial_edge_lengths = np.linalg.norm(vertices[edges_mesh[:, 0]] - vertices[edges_mesh[:, 1]], axis=-1)

    aabb = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    aabb_diag = np.linalg.norm(aabb[1] - aabb[0])
    volume_aabb = np.prod(aabb[1] - aabb[0])
    volume_stingray = compute_mesh_volume(torch.tensor(vertices), faces)
    i_edges, i_edge_flaps, i_edge_flap_corners, idx_i_edge, edges_mesh, edge_map = get_internal_edges_and_flaps(faces)

    print("Volume of the stingray: {:.2f}% of the bounding box".format(volume_stingray.item()/volume_aabb*100))
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    setting_dict = return_stingray_modes_experiment_settings(FLAGS.trial_number)
    max_to_min_weight_ratio = setting_dict['max_to_min_weight_ratio']
    init_perturb_magnitude = setting_dict['init_perturb_magnitude']
    w_fit, w_fit_cp, w_vol, w_edges, w_bend, w_g_speed = setting_dict['w_fit'], setting_dict['w_fit_cp'], setting_dict['w_vol'], setting_dict['w_edges'], setting_dict['w_bend'], setting_dict['w_g_speed']
    maxiter = setting_dict['maxiter']
    n_modes = setting_dict['n_modes']
    gt, gcp = np.array(setting_dict['gt']), np.array(setting_dict['gcp'])
    eps = setting_dict['eps']
    n_ts = setting_dict['n_ts']
    n_cp = setting_dict['n_cp']
    rho = setting_dict['rho']
    close_gait = setting_dict['close_gait']
    
    ###########################################################################
    ## DEFINE WEIGHTS
    ###########################################################################

    min_weight = 2.0 / (1.0 + max_to_min_weight_ratio)
    max_weight = max_to_min_weight_ratio * min_weight

    vertices_torch = torch.tensor(vertices, dtype=TORCH_DTYPE)
    dist_to_medial_axis = torch.abs(vertices_torch[:, 1])
    max_dist_to_medial_axis = torch.max(dist_to_medial_axis)
    scaled_dist_to_medial_axis = dist_to_medial_axis / max_dist_to_medial_axis
    weight_per_vertex = min_weight + (max_weight - min_weight) * smooth_hat_function_vary_midpoint(scaled_dist_to_medial_axis, 0.25)
    weights_surface = torch.mean(weight_per_vertex[faces], dim=1)
    weights_edges = torch.mean(weight_per_vertex[edges_mesh], dim=1)
    weights_i_edges = torch.mean(weight_per_vertex[i_edges], dim=1)
    delta_i_edges = weights_i_edges ** (1.0 / 3.0)

    ref_edge_lengths = torch.linalg.norm(vertices_torch[edges_mesh[:, 0]] - vertices_torch[edges_mesh[:, 1]], dim=1)
    ref_face_areas = compute_mesh_face_areas(vertices_torch, torch.tensor(faces))
    ref_i_edge_areas = torch.sum(ref_face_areas[i_edge_flaps], dim=1) / 3.0
    ref_edge_areas = torch.zeros(size=(edges_mesh.shape[0],), dtype=TORCH_DTYPE)
    ref_edge_areas.index_add_(0, torch.tensor(edge_map), torch.repeat_interleave(ref_face_areas / 3.0, repeats=3, dim=0))
    ref_face_normals = compute_mesh_face_normals(vertices_torch, torch.tensor(faces))
    ref_diahedral_angles = compute_mesh_diahedral_angles(ref_face_normals, i_edge_flaps)
    ref_i_edge_lengths_sq = torch.sum((vertices_torch[i_edges[:, 1]] - vertices_torch[i_edges[:, 0]])**2, dim=1)

    # Simplify the bending energy
    ref_i_edge_areas = 1.0
    ref_i_edge_lengths_sq = 1.0
    
    ###########################################################################
    ## SCALE OPTIMIZATION WEIGHTS
    ###########################################################################
    
    w_fit_scaled = w_fit / aabb_diag ** 2
    w_fit_cp_scaled = w_fit_cp / aabb_diag ** 2
    w_vol_scaled = w_vol / (volume_stingray.item() ** 2)
    w_edges_scaled = w_edges / (np.mean(initial_edge_lengths) ** 2)
    w_bend_scaled = w_bend / (1.0)
    w_g_speed_scaled = w_g_speed / (aabb_diag ** 2)
    
    ###########################################################################
    ## COMPUTE VIBRATIONAL MODES
    ###########################################################################
    
    if os.path.exists(os.path.join(path_to_output_stingray, "stingray_modes_{:02d}.json".format(FLAGS.trial_number))) and not FLAGS.recompute_modes:
        print("Modes already computed.")
        with open(os.path.join(path_to_output_stingray, "stingray_modes_{:02d}.json".format(FLAGS.trial_number)), 'r') as json_file:
            json_modes = json.load(json_file)
            lambdas = torch.tensor(json_modes['lambdas'], dtype=TORCH_DTYPE)
            modes_torch = torch.tensor(json_modes['modes'], dtype=TORCH_DTYPE)
        
    else:
        print("Computing modes...")
        total_energy = lambda verts_tmp: w_edges_scaled * edge_preservation_quad(verts_tmp.reshape(1, -1, 3), edges_mesh, ref_edge_lengths, weights=weights_edges) + w_bend_scaled * bending_small_deformation(verts_tmp.reshape(1, -1, 3), faces, i_edge_flaps, delta_i_edges, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas)
    
        mass_matrix_reduced = torch.tensor(igl.massmatrix(vertices, faces).todense())
        mass_matrix = torch.zeros(3*vertices.shape[0], 3*vertices.shape[0], dtype=TORCH_DTYPE)
        mass_matrix[0::3, 0::3] = mass_matrix_reduced
        mass_matrix[1::3, 1::3] = mass_matrix_reduced
        mass_matrix[2::3, 2::3] = mass_matrix_reduced
        mass_matrix = mass_matrix / 3.0
        
        hess_matrix = hessian(total_energy, inputs=vertices_torch.reshape(-1,))
        lambdas, modes = spl.eigh(hess_matrix.numpy(), b=mass_matrix.numpy(), subset_by_index=[0, n_modes+5])
        modes = modes[:, 6:]
        lambdas = lambdas[6:]
        modes_torch = torch.tensor(modes, dtype=TORCH_DTYPE)
        
        json_modes = {
            "lambdas": lambdas.tolist(),
            "modes": modes_torch.tolist(),
            "mass_matrix": mass_matrix_reduced.tolist(),
            "hessian_matrix": hess_matrix.tolist(),
        }
        
        with open(os.path.join(path_to_output_stingray, "stingray_modes_{:02d}.json".format(FLAGS.trial_number)), 'w') as json_file:
            json.dump(json_modes, json_file, indent=4)
            
        print("Modes computed!")
    
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
    
    cps = torch.zeros(n_cp, n_modes)
    torch.manual_seed(0)
    cps += init_perturb_magnitude * torch.randn(n_cp, n_modes)

    pos_ = elasticity_hessian_modes_generation_cubic_splines(cps, modes_torch, vertices_ref, n_ts, close_motion=close_gait)

    normals_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()
    voronoi_areas = vmap_compute_voronoi_areas(pos_, torch.tensor(faces))
    masses = rho * np.tile(voronoi_areas[0], reps=(n_ts, 1)) # use the first time step voronoi areas as masses
    
    g0 = np.zeros(shape=(7,))
    g0[3] = 1.0

    pos, tangents, g = multiple_steps_forward(
        pos_.numpy(), normals_, masses, a_weights.numpy(), b_weights.numpy(), force_0, torque_0, g0=g0,
    )

    save_path = os.path.join(path_to_output_stingray, "stingray_vibrational_init{:02d}.json".format(FLAGS.trial_number))
    export_meshes_to_json(
        pos_, g, pos, force_0, torque_0, edges_mesh, faces, save_path,
        target_final_g=gt,
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    obj_and_grad_scipy = lambda x: obj_and_grad(
        x, n_ts, n_cp, masses, a_weights.numpy(), b_weights.numpy(), 
        faces, edges_mesh, volume_stingray.item(), torch.tensor(initial_edge_lengths),
        force_0, torque_0, close_gait, gt, gcp,
        vertices_ref, modes_torch,
        i_edge_flaps, ref_diahedral_angles, ref_i_edge_lengths_sq, ref_i_edge_areas, 
        weights_edges, delta_i_edges,
        w_fit_scaled, w_fit_cp_scaled, w_vol_scaled, w_edges_scaled, w_bend_scaled, w_g_speed_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    
    params_0 = cps.reshape(-1,).numpy()
    lb = -3.0 * np.ones_like(params_0)
    ub = 3.0 * np.ones_like(params_0)
    bnds = Bounds(lb, ub)

    time_start = time.time()
    results_opt = minimize(
        obj_and_grad_scipy, params_0, bounds=bnds, jac=True, method='L-BFGS-B', 
        options={'ftol':1.0e-12, 'gtol':1.0e-7, 'disp': True, 'maxiter': maxiter},
    )
    optim_duration = time.time() - time_start
    
    ###########################################################################
    ## RUN THE OPTIMAL FORWARD PASS AND SAVE THE RESULTS
    #########################################3##################################
    
    params_torch_opt = torch.tensor(results_opt.x)

    cps_opt = params_torch_opt.reshape(n_cp, -1)
    pos_opt_ = elasticity_hessian_modes_generation_cubic_splines(cps_opt, modes_torch, vertices_ref, n_ts, close_motion=close_gait)
    pos_np_opt_ = pos_opt_.detach().numpy()

    n_cycles = 7
    pos_opt_repeat_ = np.tile(pos_opt_, reps=(n_cycles, 1, 1))
    tangents_opt_repeat_ = fun_anisotropy_dir(torch.tensor(pos_opt_repeat_)).numpy()
    masses_repeat = np.tile(masses, reps=(n_cycles, 1))
    a_weight_repeat = np.tile(a_weights, reps=(n_cycles, 1))
    b_weight_repeat = np.tile(b_weights, reps=(n_cycles, 1))
    options = {'maxfev': 2000}
    pos_opt_repeat, tangents_opt_repeat, g_opt_repeat = multiple_steps_forward(
        pos_opt_repeat_, tangents_opt_repeat_, masses_repeat, a_weight_repeat, b_weight_repeat, force_0, torque_0, options=options
    )
    
    save_path = os.path.join(path_to_output_stingray, "stingray_opt_{:02d}.json".format(FLAGS.trial_number))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_vol': w_vol,
        'w_vol_scaled': w_vol_scaled,
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
        pos_opt_repeat_, g_opt_repeat, pos_opt_repeat, force_0, torque_0, edges_mesh, faces, save_path,
        weights_optim=weights_optim, 
        quantities_per_vertex=quantities_per_vertex, quantities_per_face=quantities_per_face,
        quantities_per_edge=quantities_per_edge,
        optimization_settings=setting_dict, optimization_duration=optim_duration,
    )
    
    print("Optimization results:")
    print(g_opt_repeat[n_ts-1, 4:] - gt[4:])
    print_quaternion(torch.tensor(g[-1, :4]))
    print_quaternion(torch.tensor(g_opt_repeat[n_ts-1, :4]))
    print_quaternion(torch.tensor(gt[:4]))


if __name__ == '__main__':
    app.run(main)