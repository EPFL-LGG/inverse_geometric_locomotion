"""This script can be executed to let a snake turn in a confined space.

In a terminal, with the conda environment turned on, run the following command line:

python snake_confined.py --trial_number=0
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
path_to_output_snake = os.path.join(path_to_output, "snake_confined/")

if not os.path.exists(path_to_output_snake):
    os.makedirs(path_to_output_snake)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import numpy as np
from scipy.optimize import minimize, Bounds
import torch
import time

from geometry_io import export_snakes_to_json
from objectives import compare_last_orientation, grad_compare_last_orientation
from objectives import com_displacement_sq, grad_com_displacement_sq
from objectives import energy_path, grad_energy_path
from objectives import avoid_implicit, grad_avoid_implicit
from objectives import repulsive_curve, grad_repulsive_curve
from objectives import edge_centers_repulsion, grad_edge_centers_repulsion
from obstacle_implicits import BoxImplicit, ComplementaryImplicit
from scipy.optimize import minimize, Bounds
from snake_confined_settings import return_snake_teaser_experiment_settings
from snake_shapes import serpenoid_generation_cubic_splines_varying_wavelengths
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from utils import print_quaternion

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", None, "The trial number for the experiment.", lower_bound=0, upper_bound=3, required=True)

def obj_and_grad_params(
    params, n_steps, gt, masses, a_weights, b_weights, 
    force_0, torque_0, n_cp, snake_length, n_points_snake, close_snake_gait,
    w_fit_scaled, w_energy_scaled, w_rep_scaled, w_ecr_scaled, w_com_scaled, w_obs_scaled, 
    alpha_rep, beta_rep, n_edge_disc, edges, length_threshold, obstacle,
    fun_anisotropy_dir, fun_obj_grad_g,
):
    '''
    Args:
        params (np.ndarray of shape (n_params,)): Parameters of the optimization problem.
        n_steps (int): Number of steps of the optimization problem.
        gt (np.ndarray of shape (7,)): Target orientation.
        masses (np.ndarray of shape (n_steps, n_points_snake)): Masses of the snake.
        a_weights (np.ndarray of shape (n_steps, n_points_snake)): Weights of the snake.
        b_weights (np.ndarray of shape (n_steps, n_points_snake)): Weights of the snake.
        force_0 (np.ndarray of shape (3,)): External force applied to the snake.
        torque_0 (np.ndarray of shape (3,)): External torque applied to the snake.
        n_cp (int): Number of control points of the snake.
        snake_length (float): Length of the snake.
        n_points_snake (int): Number of points of the snake.
        close_snake_gait (bool): Whether the snake gait is closed or not.
        w_*_scaled (float): The weight of the corresponding term.
        alpha_rep: float, exponent of the repulsion kernel for the distance between points along the tangent direction
        beta_rep: float, exponent of the repulsion kernel for the distance between points
        n_edge_disc: int, number of additional discretization points along the edges (the original paper assumes n_edge_disc=0)
        edges: torch tensor of shape (n_edges, 2)
        length_threshold: float, threshold for the edge length
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the snake.
        fun_obj_grad_g (callable): Function that computes the objective and its gradient w.r.t. the orientation.
        
    Returns:
        obj (float): Objective value.
        grad_params (np.ndarray of shape (n_params,)): Gradient of the objective w.r.t. the parameters.
    '''
    
    params = torch.tensor(params, requires_grad=True)
    example_pos_ = torch.zeros(size=(n_steps, n_points_snake, 3))
    
    pos_ = serpenoid_generation_cubic_splines_varying_wavelengths(
        params.reshape(-1, 3), snake_length, 
        example_pos_, n_steps, n_points_snake, n_cp, close_snake=close_snake_gait, flip_snake=True,
    )

    pos_np_ = pos_.detach().numpy()
    tangents_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
    
    options = {'maxfev': 2000}
    pos, tangents, g = multiple_steps_forward(
        pos_np_, tangents_np_, masses, a_weights, b_weights, force_0, torque_0, options=options,
    )
    
    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, gt, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, w_rep_scaled, w_ecr_scaled, w_com_scaled, w_obs_scaled, alpha_rep, beta_rep, n_edge_disc, edges, length_threshold, obstacle, fun_anisotropy_dir)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_steps, -1)
    grad_shape_obj = multiple_steps_backward_pos_(pos_np_, pos, tangents, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir)
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_steps, n_points_snake, 3)))

    grad_params = params.grad.numpy()
    grad_params[:3] = 0.0 # The first control point is fixed
    
    return obj, grad_params

def fun_obj_grad_g(gs, pos_, gt, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, w_rep_scaled, w_ecr_scaled, w_com_scaled, w_obs_scaled, alpha_rep, beta_rep, n_edge_disc, edges, length_threshold, obstacle, fun_anisotropy_dir):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        gt: np.ndarray of shape (7,)
        masses: np.ndarray of shape (n_ts, n_points)
        a_weights: np.ndarray of shape (n_ts, n_points)
        b_weights: np.ndarray of shape (n_ts, n_points)
        w_*_scaled: float, the weight of the corresponding term
        alpha_rep: float, exponent of the repulsion kernel for the distance between points along the tangent direction
        beta_rep: float, exponent of the repulsion kernel for the distance between points
        n_edge_disc: int, number of additional discretization points along the edges (the original paper assumes n_edge_disc=0)
        edges: torch tensor of shape (n_edges, 2)
        length_threshold: float, threshold for the edge length
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the snake.
        
    Returns:
        obj: torch.tensor representing a scalar
        grad_g: torch.tensor of shape (n_ts, 7)
        grad_pos_: torch.tensor of shape (n_ts, n_points, 3)
    '''
    obj_fit = compare_last_orientation(gs, gt)
    grad_fit_g = grad_compare_last_orientation(gs, gt)
    grad_fit_pos_ = np.zeros_like(pos_)
    if w_energy_scaled == 0.0:
        obj_energy, grad_energy_g, grad_energy_pos_ = 0.0, 0.0, 0.0
    else:
        obj_energy = energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)
        grad_energy_g, grad_energy_pos_ = grad_energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)
    if w_com_scaled == 0.0:
        obj_com, grad_com_g, grad_com_pos_ = 0.0, 0.0, 0.0
    else:
        obj_com = com_displacement_sq(gs)
        grad_com_g = grad_com_displacement_sq(gs)
        grad_com_pos_ = 0.0
    if w_obs_scaled == 0.0:
        obj_obs, grad_obs_g, grad_obs_pos_ = 0.0, 0.0, 0.0
    else:
        obj_obs = avoid_implicit(pos_, gs, obstacle)
        grad_obs_g, grad_obs_pos_ = grad_avoid_implicit(pos_, gs, obstacle)
    if w_rep_scaled == 0.0:
        obj_rep, grad_rep_g, grad_rep_pos_ = 0.0, 0.0, 0.0
    else:
        obj_rep = repulsive_curve(pos_, alpha_rep, beta_rep, n_edge_disc=n_edge_disc)
        grad_rep_g, grad_rep_pos_ = grad_repulsive_curve(pos_, alpha_rep, beta_rep, n_edge_disc=n_edge_disc)
    if w_ecr_scaled == 0.0:
        obj_ecr, grad_ecr_g, grad_ecr_pos_ = 0.0, 0.0, 0.0
    else:
        obj_ecr = edge_centers_repulsion(pos_, edges, length_threshold, n_edge_disc=n_edge_disc)
        grad_ecr_g, grad_ecr_pos_ = grad_edge_centers_repulsion(pos_, edges, length_threshold, n_edge_disc=n_edge_disc)

    obj = w_fit_scaled * obj_fit + w_energy_scaled * obj_energy + w_rep_scaled * obj_rep + w_com_scaled * obj_com + w_obs_scaled * obj_obs + w_ecr_scaled * obj_ecr
    grad_g = w_fit_scaled * grad_fit_g + w_energy_scaled * grad_energy_g + w_rep_scaled * grad_rep_g + w_obs_scaled * grad_obs_g + w_com_scaled * grad_com_g + w_ecr_scaled * grad_ecr_g
    grad_pos_ = w_fit_scaled * grad_fit_pos_ + w_energy_scaled * grad_energy_pos_ + w_rep_scaled * grad_rep_pos_ + w_obs_scaled * grad_obs_pos_ + w_com_scaled * grad_com_pos_ + w_ecr_scaled * grad_ecr_pos_

    return obj, grad_g, grad_pos_

class OptimizationBookkeeper:
    def __init__(
        self, n_steps, gt, snake_length, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, n_points_snake, close_snake_gait, 
        w_fit_scaled, w_energy_scaled, w_rep_scaled, w_ecr_scaled, w_com_scaled, w_obs_scaled,
        alpha_rep, beta_rep, n_edge_disc, edges, length_threshold, obstacle,
        fun_anisotropy_dir, fun_obj_grad_g
    ):
        self.obj_values = []
        self.params_values = []
        self.g_values = []
        self.time_values = [0.0]
        self.start_time = time.time()

        self.n_steps = n_steps
        self.snake_length = snake_length
        self.masses = masses
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.force_0 = force_0
        self.torque_0 = torque_0
        self.n_cp = n_cp
        self.n_points_snake = n_points_snake
        self.close_snake_gait = close_snake_gait
        
        self.w_fit_scaled = w_fit_scaled
        self.w_energy_scaled = w_energy_scaled
        self.w_rep_scaled = w_rep_scaled
        self.w_ecr_scaled = w_ecr_scaled
        self.w_com_scaled = w_com_scaled
        self.w_obs_scaled = w_obs_scaled
        self.alpha_rep = alpha_rep
        self.beta_rep = beta_rep
        self.n_edge_disc = n_edge_disc
        self.edges = edges
        self.length_threshold = length_threshold
        self.obstacle = obstacle
        self.gt = gt
        self.fun_obj_grad_g = fun_obj_grad_g
        self.fun_anisotropy_dir = fun_anisotropy_dir
        
    def callback(self, x):
        self.time_values.append(time.time() - self.start_time)
        self.params_values.append(x.tolist())
        params = torch.tensor(x)
        example_pos_ = torch.zeros(size=(self.n_steps, self.n_points_snake, 3))
        pos_ = serpenoid_generation_cubic_splines_varying_wavelengths(
            params.reshape(-1, 3), self.snake_length,
            example_pos_, self.n_steps, self.n_points_snake, self.n_cp, close_snake=self.close_snake_gait, flip_snake=True,
        )
        pos_np_ = pos_.detach().numpy()
        tangents_np_ = self.fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
        options = {'maxfev': 2000}
        pos, tangents, g = multiple_steps_forward(
            pos_np_, tangents_np_, self.masses, self.a_weights, self.b_weights, self.force_0, self.torque_0, options=options
        )
        self.g_values.append(g.tolist())
        obj, grad_obj_g, grad_obj_pos_ = self.fun_obj_grad_g(g, pos_np_, self.gt, self.masses, self.a_weights, self.b_weights, self.w_fit_scaled, self.w_energy_scaled, self.w_rep_scaled, self.w_ecr_scaled, self.w_com_scaled, self.w_obs_scaled, self.alpha_rep, self.beta_rep, self.n_edge_disc, self.edges, self.length_threshold, self.obstacle, self.fun_anisotropy_dir)
        self.obj_values.append(obj.item())


def main(_):
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    trial_number = FLAGS.trial_number
    tag_experiment = "_{:02d}".format(trial_number)

    settings_dict = return_snake_teaser_experiment_settings(trial_number)
    n_points_snake = settings_dict["n_pts"]
    n_cp, n_ts = settings_dict["n_cp"], settings_dict["n_ts"]
    init_wavelength, max_wavelength, min_wavelength = settings_dict["init_wavelength"], settings_dict["max_wavelength"], settings_dict["min_wavelength"]
    rho, eps, snake_length, close_snake_gait = settings_dict['rho'], settings_dict['eps'], settings_dict['snake_length'], settings_dict['close_gait']
    w_fit, w_obs, w_energy, w_com, w_rep, w_ecr = settings_dict['w_fit'], settings_dict['w_obs'], settings_dict['w_energy'], settings_dict['w_com'], settings_dict['w_rep'], settings_dict['w_ecr']
    alpha_rep, beta_rep, n_edge_disc = settings_dict['alpha_rep'], settings_dict['beta_rep'], settings_dict['n_edge_disc']
    length_threshold = settings_dict['length_threshold']
    edges = torch.zeros(size=(n_points_snake-1, 2), dtype=torch.int64)
    edges[:, 0] = torch.arange(0, n_points_snake-1)
    edges[:, 1] = torch.arange(1, n_points_snake)
    gt = np.array(settings_dict['gt'])
    boxsize = settings_dict['boxsize']
    maxiter = settings_dict['maxiter']
    cps = torch.tensor(settings_dict["cps"])

    print(w_fit, w_obs, w_energy, w_com, w_rep, w_ecr)
    w_fit_scaled = w_fit / snake_length ** 2
    w_energy_scaled = w_energy / (rho * snake_length * snake_length ** 2)
    w_rep_scaled = w_rep / (2.0 * snake_length ** 2)
    w_ecr_scaled = w_ecr / (2.0 * snake_length ** 2)
    w_com_scaled = w_com / snake_length ** 2
    w_obs_scaled = w_obs / (2.0 * boxsize ** 2)

    def fun_anisotropy_dir(x):
        tangents = torch.zeros_like(x)
        tangents[..., 0, :] = x[..., 1, :] - x[..., 0, :]
        tangents[..., 1:-1, :] = x[..., 2:, :] - x[..., :-2, :]
        tangents[..., -1, :] = x[..., -1, :] - x[..., -2, :]
        tangents = tangents / torch.linalg.norm(tangents, dim=-1, keepdims=True)
        return tangents
    
    ###########################################################################
    ## GENERATE THE GEOMETRY
    ###########################################################################

    # example_pos_ serves providing the expected shape of pos_
    example_pos_ = torch.zeros(size=(n_ts, n_points_snake, 3))
    pos_ = serpenoid_generation_cubic_splines_varying_wavelengths(
        cps, snake_length, 
        example_pos_, n_ts, n_points_snake, n_cp, close_snake=close_snake_gait, flip_snake=True,
    ).numpy()

    tangents_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()

    edge_lengths = np.linalg.norm(pos_[:, 1:] - pos_[:, :-1], axis=-1)
    voronoi_cell_lengths = np.zeros(shape=(n_ts, n_points_snake))
    voronoi_cell_lengths[:, :-1] = edge_lengths / 2.0
    voronoi_cell_lengths[:, 1:] = edge_lengths / 2.0
    masses = rho * np.tile(voronoi_cell_lengths[0], reps=(n_ts, 1))
    a_weights = np.ones(shape=(n_ts, n_points_snake))
    b_weights = (eps - 1.0) * np.ones(shape=(n_ts, n_points_snake))

    force_0 = np.zeros(shape=(3,))
    torque_0 = np.zeros(shape=(3,))
        
    ###########################################################################
    ## MOTION FROM NOTHING
    ###########################################################################
    
    box_params = torch.tensor([0.0, 0.0, 0.0, boxsize, boxsize, 1.0])
    box_implicit = BoxImplicit(box_params)
    obstacle = ComplementaryImplicit(box_implicit)

    pos, tangents, g = multiple_steps_forward(
        pos_, tangents_, masses, a_weights, b_weights, force_0, torque_0, g0=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )

    save_path = os.path.join(path_to_output_snake, "snake_confined_init{}.json".format(tag_experiment))

    export_snakes_to_json(
        pos_, g, pos, force_0, torque_0, save_path, edges=None,
        weights_optim=None, quantities_per_vertex=None,
        quantities_per_edge=None, target_checkpoints_g=gt,
        
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    obj_and_grad_params_scipy = lambda x_: obj_and_grad_params(
        x_, n_ts, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, snake_length, n_points_snake, close_snake_gait, 
        w_fit_scaled, w_energy_scaled, w_rep_scaled, w_ecr_scaled, w_com_scaled, w_obs_scaled,
        alpha_rep, beta_rep, n_edge_disc, edges, length_threshold, obstacle,
        fun_anisotropy_dir, fun_obj_grad_g,
    )

    optim_bookkeeper = OptimizationBookkeeper(
        n_ts, gt, snake_length, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, n_points_snake, close_snake_gait, 
        w_fit_scaled, w_energy_scaled, w_rep_scaled, w_ecr_scaled, w_com_scaled, w_obs_scaled,
        alpha_rep, beta_rep, n_edge_disc, edges, length_threshold, obstacle,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    optim_cb = optim_bookkeeper.callback
    
    params0 = cps.reshape(-1,).numpy()
    settings_dict['params_init'] = params0.tolist()

    lb = - np.inf * np.ones_like(params0)
    lb[2::3] = min_wavelength
    ub = np.inf * np.ones_like(params0)
    ub[2::3] = max_wavelength
    bnds = Bounds(lb, ub)

    start_time = time.time()
    result = minimize(
        obj_and_grad_params_scipy, params0, bounds=bnds, jac=True, 
        method='L-BFGS-B', options={'disp': True, 'maxiter': maxiter, 'ftol': 1.0e-10, 'gtol': 1.0e-6},
        callback=optim_cb,
    )
    optim_duration = time.time() - start_time
    
    ###########################################################################
    ## RUN THE OPTIMAL FORWARD PASS AND SAVE THE RESULTS
    ###########################################################################
    
    params_opt = torch.tensor(result.x.copy())
    settings_dict['params_opt'] = params_opt.tolist()
    pos_opt_ = serpenoid_generation_cubic_splines_varying_wavelengths(
        params_opt.reshape(-1, 3), snake_length,
        example_pos_, n_ts, n_points_snake, n_cp, close_snake=close_snake_gait, flip_snake=True,
    )
    pos_opt_ = pos_opt_.detach().numpy()
    tangents_opt_ = fun_anisotropy_dir(torch.tensor(pos_opt_)).numpy()


    options = {'maxfev': 2000}
    pos_opt, tangents_opt, g_opt = multiple_steps_forward(
        pos_opt_, tangents_opt_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )

    save_path_opt = os.path.join(path_to_output_snake, "snake_confined_opt{}.json".format(tag_experiment))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_energy': w_energy,
        'w_energy_scaled': w_energy_scaled,
        'w_rep': w_rep,
        'w_rep_scaled': w_rep_scaled,
        'w_ecr': w_ecr,
        'w_ecr_scaled': w_ecr_scaled,
        'w_com': w_com,
        'w_com_scaled': w_com_scaled,
        'w_obs': w_obs,
        'w_obs_scaled': w_obs_scaled,
    }

    optim_evol_data = {
        'obj_values': optim_bookkeeper.obj_values,
        'optim_duration': optim_duration,
        'params_values': optim_bookkeeper.params_values,
        'g_values': optim_bookkeeper.g_values,
        'time_values': optim_bookkeeper.time_values,
    }

    export_snakes_to_json(
        pos_opt_, g_opt, pos_opt, force_0, torque_0, save_path_opt, edges=None,
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