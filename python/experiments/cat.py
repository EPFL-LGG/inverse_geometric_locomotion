"""This script can be executed to train a cat its self-righting gait.

In a terminal, with the conda environment turned on, run the following command line:

python cat.py --trial_number=0
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
path_to_output_cat = os.path.join(path_to_output, "cat/")

if not os.path.exists(path_to_output_cat):
    os.makedirs(path_to_output_cat)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import numpy as np
from scipy.optimize import minimize, Bounds
import torch
import time

from geometry_io import export_graphs_to_json
from objectives import compare_last_orientation, grad_compare_last_orientation
from objectives import edge_preservation_quad, grad_edge_preservation_quad
from objectives import compare_to_target_gait, grad_compare_to_target_gait
from scipy.optimize import minimize, Bounds
from cat_settings import return_cat_experiment_settings
from surface_shapes import deformations_generation_cubic_splines
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from utils import axis_angle_to_quaternion, print_quaternion, register_points_torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", None, "The trial number for the experiment.", lower_bound=0, upper_bound=1, required=True)

def obj_and_grad_params(
    params, n_steps, gt, masses, a_weights, b_weights, 
    force_0, torque_0, n_cp, close_gait,
    vertices_undeformed, vertices_ref_rot, defos_init, free_vertex_dofs,
    pos_target_, edges_graph, edge_lengths_ref,
    w_fit_scaled, w_edges_scaled, w_gait_fit_scaled,
    fun_anisotropy_dir, fun_obj_grad_g,
):
    '''
    Args:
        params (np.ndarray of shape (n_params,)): Parameters of the optimization problem.
        n_steps (int): Number of steps of the optimization problem.
        gt (np.ndarray of shape (7,)): Target orientation.
        masses (np.ndarray of shape (n_steps, n_points)): Masses.
        a_weights (np.ndarray of shape (n_steps, n_points)): Weights.
        b_weights (np.ndarray of shape (n_steps, n_points)): Weights.
        force_0 (np.ndarray of shape (3,)): External force.
        torque_0 (np.ndarray of shape (3,)): External torque.
        n_cp (int): Number of control points.
        close_gait (bool): Whether the snake gait is closed or not.
        w_*_scaled (float): The weight of the corresponding term.
        vertices_undeformed (np.ndarray of shape (n_points, 3)): Initial vertices.
        vertices_ref_rot (np.ndarray of shape (n_points, 3)): Vertices used for registration.
        defos_init (torch.tensor of shape (n_ts, n_points, 3)): Initial deformations.
        free_vertex_dofs (list): List of free vertex dofs.
        pos_target_ (torch.tensor of shape (n_ts, n_points, 3)): Target positions.
        edges_graph (np.ndarray of shape (n_edges, 2)): Edges of the graph.
        edge_lengths_ref (torch.tensor of shape (n_edges,)): Edge lengths of the reference shape.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction.
        fun_obj_grad_g (callable): Function that computes the objective and its gradient w.r.t. the orientation.
        
    Returns:
        obj (float): Objective value.
        grad_params (np.ndarray of shape (n_params,)): Gradient of the objective w.r.t. the parameters.
    '''
    
    params_torch = torch.tensor(params)
    params_torch.requires_grad = True
    
    cps_tmp = params_torch.reshape(n_cp, -1)
    n_free_dofs = cps_tmp.shape[1]
    pos_ = deformations_generation_cubic_splines(
        cps_tmp, torch.tensor(vertices_undeformed), torch.tensor(vertices_ref_rot), defos_init, free_vertex_dofs, n_steps, close_motion=close_gait,
    )
    pos_np_ = pos_.detach().numpy()
    normals_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
    
    options = {'maxfev': 2000}
    pos, normals, g = multiple_steps_forward(
        pos_np_, normals_np_, masses, a_weights, b_weights, force_0, torque_0, options=options,
    )
    
    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, gt, pos_target_, edges_graph, edge_lengths_ref, w_fit_scaled, w_edges_scaled, w_gait_fit_scaled)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_steps, -1)
    grad_shape_obj = multiple_steps_backward_pos_(pos_np_, pos, normals, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir)
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_steps, -1, 3)))
    grad_params_obj = params_torch.grad.numpy()
    grad_params_obj[:n_free_dofs] = 0.0 # First shape is fixed
    grad_params_obj[-n_free_dofs:] = 0.0 # Last shape is fixed
    
    return obj, grad_params_obj

def fun_obj_grad_g(gs, pos_, gt, pos_target_, edges_graph, edge_lengths_ref, w_fit_scaled, w_edges_scaled, w_gait_fit_scaled):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        gt: np.ndarray of shape (7,)
        w_*_scaled: float, the weight of the corresponding term
        
    Returns:
        obj: torch.tensor representing a scalar
        grad_g: torch.tensor of shape (n_ts, 7)
        grad_pos_: torch.tensor of shape (n_ts, n_points, 3)
    '''
    obj_fit = compare_last_orientation(gs, gt)
    grad_fit_g = grad_compare_last_orientation(gs, gt)
    grad_fit_pos_ = np.zeros_like(pos_)

    if w_edges_scaled == 0.0:
        obj_edges, grad_edges_g, grad_edges_pos_ = 0.0, 0.0, 0.0
    else:
        obj_edges = edge_preservation_quad(pos_, edges_graph, edge_lengths_ref)
        grad_edges_g, grad_edges_pos_ = grad_edge_preservation_quad(pos_, edges_graph, edge_lengths_ref)

    if w_gait_fit_scaled == 0.0:
        obj_gait, grad_gait_g, grad_gait_pos_ = 0.0, 0.0, 0.0
    else:
        obj_gait = compare_to_target_gait(pos_, pos_target_)
        grad_gait_g, grad_gait_pos_ = grad_compare_to_target_gait(pos_, pos_target_)

    obj = w_fit_scaled * obj_fit + w_edges_scaled * obj_edges + w_gait_fit_scaled * obj_gait
    grad_g = w_fit_scaled * grad_fit_g + w_edges_scaled * grad_edges_g + w_gait_fit_scaled * grad_gait_g
    grad_pos_ =  w_fit_scaled * grad_fit_pos_ + w_edges_scaled * grad_edges_pos_ + w_gait_fit_scaled * grad_gait_pos_

    return obj, grad_g, grad_pos_

class OptimizationBookkeeper:
    def __init__(
        self, n_steps, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, close_gait,
        vertices_undeformed, vertices_ref_rot, defos_init, free_vertex_dofs,
        pos_target_, edges_graph, edge_lengths_ref,
        w_fit_scaled, w_edges_scaled, w_gait_fit_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    ):
        self.obj_values = []
        self.params_values = []
        self.g_values = []
        self.time_values = [0.0]
        self.start_time = time.time()

        self.n_steps = n_steps
        self.vertices_undeformed = vertices_undeformed
        self.vertices_ref_rot = vertices_ref_rot
        self.defos_init = defos_init
        self.free_vertex_dofs = free_vertex_dofs
        self.pos_target_ = pos_target_
        self.edges_graph = edges_graph
        self.edge_lengths_ref = edge_lengths_ref
        self.masses = masses
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.force_0 = force_0
        self.torque_0 = torque_0
        self.n_cp = n_cp
        self.close_gait = close_gait
        
        self.w_fit_scaled = w_fit_scaled
        self.w_edges_scaled = w_edges_scaled
        self.w_gait_fit_scaled = w_gait_fit_scaled
        self.gt = gt
        self.fun_obj_grad_g = fun_obj_grad_g
        self.fun_anisotropy_dir = fun_anisotropy_dir
        
    def callback(self, x):
        self.time_values.append(time.time() - self.start_time)
        self.params_values.append(x.tolist())
        params = torch.tensor(x)
        cps_tmp = params.reshape(self.n_cp, -1)
        pos_ = deformations_generation_cubic_splines(
            cps_tmp, torch.tensor(self.vertices_undeformed), torch.tensor(self.vertices_ref_rot), self.defos_init, self.free_vertex_dofs, self.n_steps, close_motion=self.close_gait,
        )
        pos_np_ = pos_.detach().numpy()
        normals_np_ = self.fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
        
        options = {'maxfev': 2000}
        pos, normals, g = multiple_steps_forward(
            pos_np_, normals_np_, self.masses, self.a_weights, self.b_weights, self.force_0, self.torque_0, options=options,
        )
        self.g_values.append(g.tolist())
        obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, self.gt, self.pos_target_, self.edges_graph, self.edge_lengths_ref, self.w_fit_scaled, self.w_edges_scaled, self.w_gait_fit_scaled)
        self.obj_values.append(obj.item())


def main(_):
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    trial_number = FLAGS.trial_number
    tag_experiment = "_{:02d}".format(trial_number)

    settings_dict = return_cat_experiment_settings(trial_number)
    n_points = settings_dict["n_pts"]
    n_cp, n_ts = settings_dict["n_cp"], settings_dict["n_ts"]
    rho, eps, close_gait = settings_dict['rho'], settings_dict['eps'], settings_dict['close_gait']
    w_fit, w_edges, w_gait_fit = settings_dict['w_fit'], settings_dict['w_edges'], settings_dict['w_gait_fit']
    gt = np.array(settings_dict['gt'])
    maxiter = settings_dict['maxiter']
    
    vertices, vertices_ref, vertices_perturbed = np.array(settings_dict['vertices']), np.array(settings_dict['vertices_ref']), np.array(settings_dict['vertices_perturbed'])
    initial_edge_lengths = torch.tensor(settings_dict['initial_edge_lengths'])
    cps = torch.tensor(settings_dict['cps'])
    edges_graph = np.array(settings_dict['edges_graph'])
    pos_files_ = np.array(settings_dict['pos_files_'])
    ids_subsample = np.linspace(0, pos_files_.shape[0]-1, n_ts, dtype=np.int32)
    pos_ref_ = pos_files_[ids_subsample]
    vertices_ref_rot_rep = torch.tensor(vertices_perturbed).unsqueeze(0).repeat(n_ts, 1, 1)
    pos_ref_ = register_points_torch(torch.tensor(pos_ref_), vertices_ref_rot_rep, allow_flip=False).numpy()
    aabb = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    aabb_diag = np.linalg.norm(aabb[1] - aabb[0])
    
    ids_subsample = np.linspace(0, len(settings_dict['pos_files_'])-1, settings_dict['n_cp'], dtype=np.int32)

    w_fit_scaled = w_fit / (aabb_diag ** 2)
    w_edges_scaled = w_edges / (torch.mean(initial_edge_lengths).item() ** 2)
    w_gait_fit_scaled = w_gait_fit / aabb_diag ** 2

    
    def fun_anisotropy_dir(pos_):
        '''
        Args:
            pos_: torch.tensor of shape (n_ts, n_points, 3) or (n_points, 3)
            
        Returns:
            normals_: torch.tensor of shape (n_ts, n_points, 3) or (n_points, 3)
        '''
        normals_ = torch.zeros_like(pos_)
        normals_[..., 2] = 1.0
        return normals_
    
    ###########################################################################
    ## GENERATE THE GEOMETRY
    ###########################################################################

    diss_weight = np.ones(shape=(n_ts, n_points))
    a_weights = eps * diss_weight
    b_weights = (1.0 - eps) * diss_weight

    force_0 = np.zeros(shape=(3,))
    torque_0 = np.zeros(shape=(3,))

    force_0 = np.zeros(shape=(3,))
    torque_0 = np.zeros(shape=(3,))
        
    ###########################################################################
    ## MOTION FROM NOTHING
    ###########################################################################
    defos_init = torch.zeros(size=(n_ts, n_points, 3))
    fixed_vertex_dofs = []
    free_vertex_dofs = np.setdiff1d(list(range(3*n_points)), fixed_vertex_dofs).tolist()

    pos_ = deformations_generation_cubic_splines(
        cps, torch.tensor(vertices), torch.tensor(vertices_perturbed), defos_init, free_vertex_dofs, n_ts, close_motion=close_gait,
    )

    normals_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()
    masses = rho * np.ones(shape=(n_ts, n_points))

    pos, normals, g = multiple_steps_forward(
        pos_.numpy(), normals_, masses, a_weights, b_weights, force_0, torque_0, g0=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    
    print_quaternion(torch.tensor(g[-1, :4]))

    save_path = os.path.join(path_to_output_cat, "cat_init{}.json".format(tag_experiment))
    export_graphs_to_json(
        pos_, g, pos, force_0, torque_0, edges_graph, save_path,
        weights_optim=None, 
        target_final_g=gt,
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    obj_and_grad_scipy = lambda x: obj_and_grad_params(
        x, n_ts, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, close_gait,
        vertices, vertices_perturbed, defos_init, free_vertex_dofs,
        pos_ref_, edges_graph, initial_edge_lengths,
        w_fit_scaled, w_edges_scaled, w_gait_fit_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    
    optim_bookkeeper = OptimizationBookkeeper(
        n_ts, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, close_gait,
        vertices, vertices_perturbed, defos_init, free_vertex_dofs,
        pos_ref_, edges_graph, initial_edge_lengths,
        w_fit_scaled, w_edges_scaled, w_gait_fit_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    optim_cb = optim_bookkeeper.callback
    
    params0 = cps.reshape(-1,).numpy()
    settings_dict['params_init'] = params0.tolist()
    bnds = Bounds()

    start_time = time.time()
    result = minimize(
        obj_and_grad_scipy, params0, bounds=bnds, jac=True, 
        method='L-BFGS-B', options={'disp': True, 'maxiter': maxiter, 'ftol': 1.0e-10, 'gtol': 1.0e-6},
        callback=optim_cb,
    )
    optim_duration = time.time() - start_time
    
    ###########################################################################
    ## RUN THE OPTIMAL FORWARD PASS AND SAVE THE RESULTS
    ###########################################################################
    
    params_opt = torch.tensor(result.x)
    settings_dict['params_opt'] = params_opt.tolist()
    cps_opt = params_opt.reshape(n_cp, -1)
    pos_opt_ = deformations_generation_cubic_splines(
        cps_opt, torch.tensor(vertices), torch.tensor(vertices_perturbed), defos_init, free_vertex_dofs, n_ts, close_motion=close_gait,
    )

    pos_np_opt_ = pos_opt_.detach().numpy()
    normals_np_opt_ = fun_anisotropy_dir(torch.tensor(pos_np_opt_)).numpy()

    options = {"maxfev": 2000}
    pos_opt, normals_opt, g_opt = multiple_steps_forward(
        pos_np_opt_, normals_np_opt_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )

    save_path_opt = os.path.join(path_to_output_cat, "cat_opt{}.json".format(tag_experiment))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_edges': w_edges,
        'w_edges_scaled': w_edges_scaled,
        'w_gait_fit': w_gait_fit,
        'w_gait_fit_scaled': w_gait_fit_scaled,
    }

    optim_evol_data = {
        'obj_values': optim_bookkeeper.obj_values,
        'optim_duration': optim_duration,
        'params_values': optim_bookkeeper.params_values,
        'g_values': optim_bookkeeper.g_values,
        'time_values': optim_bookkeeper.time_values,
    }

    export_graphs_to_json(
        pos_opt_, g_opt, pos_opt, force_0, torque_0, edges_graph, save_path_opt,
        weights_optim=weights_optim, quantities_per_vertex=None,
        quantities_per_edge=None,
        optimization_settings=settings_dict, optimization_duration=optim_duration,
        optimization_evolution=optim_evol_data,
    )

    
    print("Optimization results:")
    print_quaternion(torch.tensor(g[-1, :4]))
    print_quaternion(torch.tensor(g_opt[-1, :4]))
    print_quaternion(torch.tensor(gt[:4]))


if __name__ == '__main__':
    app.run(main)