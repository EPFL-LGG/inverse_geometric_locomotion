"""This script can be executed to train a snake to follow a trajectory while avoiding obstacles.

In a terminal, with the conda environment turned on, run the following command line:

python snake_ff_obstacle.py --trial_number=0
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
path_to_output_snake = os.path.join(path_to_output, "snake_ff_obstacle/")

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
from objectives import compare_last_translation, grad_compare_last_translation
from objectives import energy_path, grad_energy_path
from objectives import avoid_implicit, grad_avoid_implicit
from obstacle_implicits import SphereSquareImplicit
from scipy.optimize import minimize, Bounds
from snake_ff_obstacle_settings import return_snake_ff_obstacle_experiment_settings
from snake_shapes import snake_angles_generation_cubic_splines
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from utils import print_quaternion

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", None, "The trial number for the experiment.", lower_bound=0, upper_bound=1, required=True)

def obj_and_grad_params(
    params, n_steps, gt, masses, a_weights, b_weights,
    broken_joint_ids, broken_joint_angles,
    force_0, torque_0, n_cp, snake_length, n_points_snake, close_snake_gait,
    w_fit_scaled, w_energy_scaled, w_obs_scaled, obstacle,
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
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the snake.
        fun_obj_grad_g (callable): Function that computes the objective and its gradient w.r.t. the orientation.
        
    Returns:
        obj (float): Objective value.
        grad_params (np.ndarray of shape (n_params,)): Gradient of the objective w.r.t. the parameters.
    '''
    
    params = torch.tensor(params, requires_grad=True)
    example_pos_ = torch.zeros(size=(n_steps, n_points_snake, 3))
    
    pos_, disc_angles = snake_angles_generation_cubic_splines(
        params.reshape(n_cp, -1), snake_length, broken_joint_ids, broken_joint_angles,
        example_pos_, n_steps, n_cp, close_snake=close_snake_gait, return_discretized=True
    )

    pos_np_ = pos_.detach().numpy()
    tangents_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
    
    options = {'maxfev': 2000}
    pos, tangents, g = multiple_steps_forward(
        pos_np_, tangents_np_, masses, a_weights, b_weights, force_0, torque_0, options=options,
    )
    
    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, gt, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, w_obs_scaled, obstacle, fun_anisotropy_dir)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_steps, -1)
    grad_shape_obj = multiple_steps_backward_pos_(pos_np_, pos, tangents, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir)
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_steps, n_points_snake, 3)))
    
    return obj, params.grad.numpy()

def fun_obj_grad_g(gs, pos_, gt, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, w_obs_scaled, obstacle, fun_anisotropy_dir):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        gt: np.ndarray of shape (7,)
        masses: np.ndarray of shape (n_ts, n_points)
        a_weights: np.ndarray of shape (n_ts, n_points)
        b_weights: np.ndarray of shape (n_ts, n_points)
        w_*_scaled: float, the weight of the corresponding term
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the snake.
        
    Returns:
        obj: torch.tensor representing a scalar
        grad_g: torch.tensor of shape (n_ts, 7)
        grad_pos_: torch.tensor of shape (n_ts, n_points, 3)
    '''
    obj_fit = compare_last_translation(gs, gt)
    grad_fit_g = grad_compare_last_translation(gs, gt)
    grad_fit_pos_ = np.zeros_like(pos_)

    if w_energy_scaled == 0.0:
        obj_energy, grad_energy_g, grad_energy_pos_ = 0.0, 0.0, 0.0
    else:
        obj_energy = energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)
        grad_energy_g, grad_energy_pos_ = grad_energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)

    if w_obs_scaled == 0.0:
        obj_obs, grad_obs_g, grad_obs_pos_ = 0.0, 0.0, 0.0
    else:
        obj_obs = avoid_implicit(pos_, gs, obstacle)
        grad_obs_g, grad_obs_pos_ = grad_avoid_implicit(pos_, gs, obstacle)

    obj = w_fit_scaled * obj_fit + w_energy_scaled * obj_energy + w_obs_scaled * obj_obs
    grad_g = w_fit_scaled * grad_fit_g + w_energy_scaled * grad_energy_g + w_obs_scaled * grad_obs_g
    grad_pos_ = w_fit_scaled * grad_fit_pos_ + w_energy_scaled * grad_energy_pos_ + w_obs_scaled * grad_obs_pos_

    return obj, grad_g, grad_pos_

class OptimizationBookkeeper:
    def __init__(
        self, n_steps, gt, snake_length, masses, a_weights, b_weights, 
        broken_joint_ids, broken_joint_angles,
        force_0, torque_0, n_cp, n_points_snake, close_snake_gait, 
        w_fit_scaled, w_energy_scaled, w_obs_scaled, obstacle,
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
        self.broken_joint_ids = broken_joint_ids
        self.broken_joint_angles = broken_joint_angles
        self.force_0 = force_0
        self.torque_0 = torque_0
        self.n_cp = n_cp
        self.n_points_snake = n_points_snake
        self.close_snake_gait = close_snake_gait
        
        self.w_fit_scaled = w_fit_scaled
        self.w_energy_scaled = w_energy_scaled
        self.w_obs_scaled = w_obs_scaled
        self.obstacle = obstacle
        self.gt = gt
        self.fun_obj_grad_g = fun_obj_grad_g
        self.fun_anisotropy_dir = fun_anisotropy_dir
        
    def callback(self, x):
        self.time_values.append(time.time() - self.start_time)
        self.params_values.append(x.tolist())
        params = torch.tensor(x)
        example_pos_ = torch.zeros(size=(self.n_steps, self.n_points_snake, 3))
        pos_, disc_angles = snake_angles_generation_cubic_splines(
            params.reshape(self.n_cp, -1), self.snake_length, self.broken_joint_ids, self.broken_joint_angles,
            example_pos_, self.n_steps, self.n_cp, close_snake=self.close_snake_gait, return_discretized=True
        )
        pos_np_ = pos_.detach().numpy()
        tangents_np_ = self.fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
        options = {'maxfev': 2000}
        pos, tangents, g = multiple_steps_forward(
            pos_np_, tangents_np_, self.masses, self.a_weights, self.b_weights, self.force_0, self.torque_0, options=options
        )
        self.g_values.append(g.tolist())
        obj, grad_obj_g, grad_obj_pos_ = self.fun_obj_grad_g(g, pos_np_, self.gt, self.masses, self.a_weights, self.b_weights, self.w_fit_scaled, self.w_energy_scaled, self.w_obs_scaled, self.obstacle, self.fun_anisotropy_dir)
        self.obj_values.append(obj.item())


def main(_):
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    trial_number = FLAGS.trial_number
    tag_experiment = "_{:02d}".format(trial_number)

    settings_dict = return_snake_ff_obstacle_experiment_settings(trial_number)
    n_points_snake, n_angles = settings_dict["n_pts"], settings_dict["n_angles"]
    n_cp, n_ts = settings_dict["n_cp"], settings_dict["n_ts"]
    rho, eps, snake_length, close_snake_gait = settings_dict['rho'], settings_dict['eps'], settings_dict['snake_length'], settings_dict['close_gait']
    broken_joint_ids, broken_joint_angles = settings_dict['broken_joint_ids'], torch.tensor(settings_dict['broken_joint_angles'])
    w_fit, w_obs, w_energy = settings_dict['w_fit'], settings_dict['w_obs'], settings_dict['w_energy']
    gt = np.array(settings_dict['gt'])
    print(gt)
    # assert 0
    init_perturb_magnitude = settings_dict['init_perturb_magnitude']
    min_turning_angle, max_turning_angle = settings_dict['min_turning_angle'], settings_dict['max_turning_angle']
    maxiter = settings_dict['maxiter']
    obstacle_params = torch.tensor(settings_dict['obstacle_params'])
    n_op_angles = n_angles - len(broken_joint_ids)

    total_mass = rho * snake_length
    w_fit_scaled = w_fit / (snake_length ** 2)
    w_obs_scaled = w_obs / (2.0 * snake_length ** 2)
    w_energy_scaled = w_energy / (total_mass * snake_length ** 2)

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

    cps = torch.zeros(n_cp, n_op_angles)
    torch.manual_seed(0)
    cps += init_perturb_magnitude * torch.randn(n_cp, n_op_angles)

    example_pos_ = torch.zeros(size=(n_ts, n_points_snake, 3))
    pos_ = snake_angles_generation_cubic_splines(
        cps, snake_length, broken_joint_ids, broken_joint_angles,
        example_pos_, n_ts, n_cp, close_snake=close_snake_gait
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

    obstacle = SphereSquareImplicit(obstacle_params)

    pos, tangents, g = multiple_steps_forward(
        pos_, tangents_, masses, a_weights, b_weights, force_0, torque_0, g0=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )

    save_path = os.path.join(path_to_output_snake, "snake_ff_obstacle_init{}.json".format(tag_experiment))

    export_snakes_to_json(
        pos_, g, pos, force_0, torque_0, save_path, edges=None,
        weights_optim=None, quantities_per_vertex=None,
        quantities_per_edge=None, target_final_g=gt,
        
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    obj_and_grad_params_scipy = lambda x_: obj_and_grad_params(
        x_, n_ts, gt, masses, a_weights, b_weights,
        broken_joint_ids, broken_joint_angles,
        force_0, torque_0, n_cp, snake_length, n_points_snake, close_snake_gait, 
        w_fit_scaled, w_energy_scaled, w_obs_scaled, obstacle,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    
    optim_bookkeeper = OptimizationBookkeeper(
        n_ts, gt, snake_length, masses, a_weights, b_weights,
        broken_joint_ids, broken_joint_angles,
        force_0, torque_0, n_cp, n_points_snake, close_snake_gait, 
        w_fit_scaled, w_energy_scaled, w_obs_scaled, obstacle,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    optim_cb = optim_bookkeeper.callback
    
    params0 = cps.reshape(-1,).numpy()
    settings_dict['params_init'] = params0.tolist()

    lb = min_turning_angle * np.ones_like(params0)
    ub = max_turning_angle * np.ones_like(params0)
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
    pos_opt_ = snake_angles_generation_cubic_splines(
        params_opt.reshape(n_cp, -1), snake_length, broken_joint_ids, broken_joint_angles,
        example_pos_, n_ts, n_cp, close_snake=close_snake_gait, return_discretized=False
    )
    pos_opt_ = pos_opt_.detach().numpy()
    tangents_opt_ = fun_anisotropy_dir(torch.tensor(pos_opt_)).numpy()


    options = {'maxfev': 2000}
    pos_opt, tangents_opt, g_opt = multiple_steps_forward(
        pos_opt_, tangents_opt_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )

    save_path_opt = os.path.join(path_to_output_snake, "snake_ff_obstacle_opt{}.json".format(tag_experiment))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_energy': w_energy,
        'w_energy_scaled': w_energy_scaled,
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
        quantities_per_edge=None, target_final_g=gt,
        optimization_settings=settings_dict, optimization_duration=optim_duration,
        optimization_evolution=optim_evol_data,
    )

    
    print("Optimization results:")
    print_quaternion(torch.tensor(g[-1, :4]))
    print_quaternion(torch.tensor(g_opt[-1, :4]))
    print_quaternion(torch.tensor(gt[:4]))


if __name__ == '__main__':
    app.run(main)