"""This script can be executed to coordinate two clams attached to each other swimming in a viscous fluid.

In a terminal, with the conda environment turned on, run the following command line:

python two_clams.py --trial_number=0
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
path_to_output_two_clams = os.path.join(path_to_output, "two_clams/")

if not os.path.exists(path_to_output_two_clams):
    os.makedirs(path_to_output_two_clams)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import numpy as np
from scipy.optimize import minimize, Bounds
import torch
import time

from geometry_io import export_snakes_to_json
from objectives import displacement, grad_displacement
from objectives import energy_path, grad_energy_path
from scipy.optimize import minimize, Bounds
from two_clams_settings import return_two_clams_experiment_settings
from purcell_shapes import symmetric_purcell_generator_euclidean_cubic_splines, symmetric_purcell_generate_fun_anisotropy_dir
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from utils import print_quaternion

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", 0, "The trial number for the experiment.")

def obj_and_grad_params(
    params, n_steps, target_norm, disp_direction, masses, a_weights, b_weights, 
    force_0, torque_0, n_cp, n_pts_per_segment, n_points, close_gait,
    e1bye2, e3bye2, total_length,
    w_fit_scaled, w_energy_scaled,
    fun_anisotropy_dir, fun_obj_grad_g,
):
    '''
    Args:
        params (np.ndarray of shape (n_params,)): Parameters of the optimization problem.
        n_steps (int): Number of steps of the optimization problem.
        target_norm (float): Target displacement norm.
        disp_direction (None or np.ndarray of shape (3,)): Target displacement direction.
        masses (np.ndarray of shape (n_steps, n_points_snake)): Masses of the object.
        a_weights (np.ndarray of shape (n_steps, n_points)): Weights of the object.
        b_weights (np.ndarray of shape (n_steps, n_points)): Weights of the object.
        force_0 (np.ndarray of shape (3,)): External force applied to the object.
        torque_0 (np.ndarray of shape (3,)): External torque applied to the object.
        n_cp (int): Number of control points.
        n_pts_per_segment (int): Number of points per segment of the object.
        n_points (int): Number of points of the object.
        close_gait (bool): Whether the gait is closed or not.
        e1bye2 (float): ratio of the length of segment 1 to segment 2.
        e3bye2 (float): ratio of the length of segment 3 to segment 2.
        total_length (float): Total length of the two_clams swimmer.
        w_*_scaled (float): The weight of the corresponding term.
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the object.
        fun_obj_grad_g (callable): Function that computes the objective and its gradient w.r.t. the orientation.
        
    Returns:
        obj (float): Objective value.
        grad_params (np.ndarray of shape (n_params,)): Gradient of the objective w.r.t. the parameters.
    '''
    
    params = torch.tensor(params, requires_grad=True)
    cps_tmp = params.reshape(n_cp, 2)
    pos_ = symmetric_purcell_generator_euclidean_cubic_splines(
        cps_tmp, n_steps, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait
    )

    pos_np_ = pos_.detach().numpy()
    tangents_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
    
    options = {'maxfev': 2000}
    pos, tangents, g = multiple_steps_forward(
        pos_np_, tangents_np_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )

    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, target_norm, disp_direction, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, fun_anisotropy_dir)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_steps, -1)
    grad_shape_obj = multiple_steps_backward_pos_(pos_, pos, tangents, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir)
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_steps, n_points, 3)))

    grad_obj_params = params.grad.numpy()

    return obj, grad_obj_params

def fun_obj_grad_g(gs, pos_, target_norm, disp_direction, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, fun_anisotropy_dir):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        target_norm: float, the target displacement norm
        disp_direction: None or np.ndarray of shape (3,), the target displacement direction
        masses: np.ndarray of shape (n_ts, n_points)
        a_weights: np.ndarray of shape (n_ts, n_points)
        b_weights: np.ndarray of shape (n_ts, n_points)
        w_*_scaled: float, the weight of the corresponding term
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the object.
        
    Returns:
        obj: torch.tensor representing a scalar
        grad_g: torch.tensor of shape (n_ts, 7)
        grad_pos_: torch.tensor of shape (n_ts, n_points, 3)
    '''
    obj_fit = displacement(gs, target_norm, direction=disp_direction)
    grad_fit_g = grad_displacement(gs, target_norm, direction=disp_direction)
    grad_fit_pos_ = np.zeros_like(pos_)
    
    if w_energy_scaled == 0.0:
        obj_energy, grad_energy_g, grad_energy_pos_ = 0.0, 0.0, 0.0
    else:
        obj_energy = energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)
        grad_energy_g, grad_energy_pos_ = grad_energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)

    obj = w_fit_scaled * obj_fit + w_energy_scaled * obj_energy
    grad_g = w_fit_scaled * grad_fit_g + w_energy_scaled * grad_energy_g
    grad_pos_ = w_fit_scaled * grad_fit_pos_ + w_energy_scaled * grad_energy_pos_

    return obj, grad_g, grad_pos_

class OptimizationBookkeeper:
    def __init__(
        self, n_steps, target_norm, disp_direction, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, n_pts_per_segment, n_points, close_gait,
        e1bye2, e3bye2, total_length,
        w_fit_scaled, w_energy_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    ):
        self.obj_values = []
        self.params_values = []
        self.g_values = []
        self.time_values = [0.0]
        self.start_time = time.time()

        self.n_steps = n_steps
        self.n_pts_per_segment = n_pts_per_segment
        self.masses = masses
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.force_0 = force_0
        self.torque_0 = torque_0
        self.n_cp = n_cp
        self.n_points = n_points
        self.close_gait = close_gait

        self.e1bye2 = e1bye2
        self.e3bye2 = e3bye2
        self.total_length = total_length
        
        self.w_fit_scaled = w_fit_scaled
        self.w_energy_scaled = w_energy_scaled
        self.target_norm = target_norm
        self.disp_direction = disp_direction
        self.fun_obj_grad_g = fun_obj_grad_g
        self.fun_anisotropy_dir = fun_anisotropy_dir
        
    def callback(self, x):
        self.time_values.append(time.time() - self.start_time)
        self.params_values.append(x.tolist())
        params = torch.tensor(x)
        cps_tmp = params.reshape(self.n_cp, 2)
        pos_ = symmetric_purcell_generator_euclidean_cubic_splines(
            cps_tmp, self.n_steps, self.n_pts_per_segment, e1bye2=self.e1bye2, e3bye2=self.e3bye2, total_length=self.total_length, close_gait=self.close_gait
        )

        pos_np_ = pos_.detach().numpy()
        tangents_np_ = self.fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
        options = {'maxfev': 2000}
        pos, tangents, g = multiple_steps_forward(
            pos_np_, tangents_np_, self.masses, self.a_weights, self.b_weights, self.force_0, self.torque_0, options=options
        )
        self.g_values.append(g.tolist())
        obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, self.target_norm, self.disp_direction, self.masses, self.a_weights, self.b_weights, self.w_fit_scaled, self.w_energy_scaled, self.fun_anisotropy_dir)
        self.obj_values.append(obj.item())


def main(_):
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    trial_number = FLAGS.trial_number
    tag_experiment = "_{:02d}".format(trial_number)

    settings_dict = return_two_clams_experiment_settings(trial_number)
    n_pts_per_segment, n_cp, n_ts = settings_dict["n_pts_per_segment"], settings_dict["n_cp"], settings_dict["n_ts"]
    n_hinges = 6
    n_points = n_hinges + 5 * n_pts_per_segment
    close_gait, eps, rho = settings_dict["close_gait"], settings_dict["eps"], np.array(settings_dict["rho"])
    e1bye2, e3bye2, total_length = settings_dict["e1bye2"], settings_dict["e2bye3"], settings_dict["total_length"]
    min_abs_angles, max_abs_angles = settings_dict["min_abs_angles"], settings_dict["max_abs_angles"]
    maxiter = settings_dict['maxiter']
    init_perturb_magnitude = settings_dict['init_perturb_magnitude']
    w_fit, w_energy = settings_dict['w_fit'], settings_dict['w_energy']
    target_norm, disp_direction = settings_dict['target_disp_norm'], settings_dict['target_disp_direction']
    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5]])

    w_fit_scaled = w_fit / (total_length ** 2)
    w_energy_scaled = w_energy / (rho * total_length * total_length ** 2)

    fun_anisotropy_dir = symmetric_purcell_generate_fun_anisotropy_dir(n_pts_per_segment, edges)
    
    ###########################################################################
    ## GENERATE THE GEOMETRY
    ###########################################################################

    center = torch.tensor([0.25 * torch.pi, 0.25 * torch.pi])
    radii = 0.2 * torch.pi * torch.ones(n_cp)
    phis = torch.linspace(0.0, 1.0, n_cp + 1)[:-1]

    # Initialize alphas with fixed polar angles that vary over time
    params = torch.zeros(n_cp, 2)
    params[:, 0] = radii * torch.cos(2.0 * torch.pi * phis)
    params[:, 1] = radii * torch.sin(2.0 * torch.pi * phis)
    torch.manual_seed(0)
    params = params + center + init_perturb_magnitude * torch.randn_like(params)

    pos_ = symmetric_purcell_generator_euclidean_cubic_splines(
        params, n_ts, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait
    ).numpy()

    tangents_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()

    edge_lengths = np.linalg.norm(pos_[:, 1:] - pos_[:, :-1], axis=-1)
    voronoi_cell_lengths = np.zeros(shape=(n_ts, n_points))
    voronoi_cell_lengths[:, :-1] = edge_lengths / 2.0
    voronoi_cell_lengths[:, 1:] = edge_lengths / 2.0
    masses = rho * np.tile(voronoi_cell_lengths[0], reps=(n_ts, 1))
    a_weights = np.ones(shape=(n_ts, n_points))
    b_weights = (eps - 1.0) * np.ones(shape=(n_ts, n_points))

    force_0 = np.zeros(shape=(3,))
    torque_0 = np.zeros(shape=(3,))
        
    ###########################################################################
    ## MOTION FROM NOTHING
    ###########################################################################
    
    pos, tangents, g = multiple_steps_forward(
        pos_, tangents_, masses, a_weights, b_weights, force_0, torque_0
    )

    save_path = os.path.join(path_to_output_two_clams, "two_clams_init.json")

    export_snakes_to_json(
        pos_, g, pos, force_0, torque_0, save_path, edges=None,
        weights_optim=None, quantities_per_vertex=None,
        quantities_per_edge=None,
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    obj_and_grad_params_scipy = lambda x_: obj_and_grad_params(
        x_, n_ts, target_norm, disp_direction, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, n_pts_per_segment, n_points, close_gait,
        e1bye2, e3bye2, total_length,
        w_fit_scaled, w_energy_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    
    optim_bookkeeper = OptimizationBookkeeper(
        n_ts, target_norm, disp_direction, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, n_pts_per_segment, n_points, close_gait,
        e1bye2, e3bye2, total_length,
        w_fit_scaled, w_energy_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    optim_cb = optim_bookkeeper.callback
    
    params0 = params.reshape(-1,).numpy()
    settings_dict['params_init'] = params0.tolist()

    lb = min_abs_angles * np.ones_like(params0)
    ub = max_abs_angles * np.ones_like(params0)
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
    cps_opt = params_opt.reshape(n_cp, 2)
    pos_opt_ = symmetric_purcell_generator_euclidean_cubic_splines(
        cps_opt, n_ts, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait
    ).numpy()
    tangents_opt_ = fun_anisotropy_dir(torch.tensor(pos_opt_)).numpy()
    options = {'maxfev': 2000}
    pos_opt, tangents_opt, g_opt = multiple_steps_forward(
        pos_opt_, tangents_opt_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )

    save_path_opt = os.path.join(path_to_output_two_clams, "two_clams_opt{}.json".format(tag_experiment))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_energy': w_energy,
        'w_energy_scaled': w_energy_scaled,
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
    print("Final position:", g_opt[-1, 4:])
    print_quaternion(torch.tensor(g[-1, :4]))
    print_quaternion(torch.tensor(g_opt[-1, :4]))


if __name__ == '__main__':
    app.run(main)