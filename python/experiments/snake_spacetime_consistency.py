"""This script can be executed to explore the effect of space and time discretization.

In a terminal, with the conda environment turned on, run the following command line:

python snake_spacetime_consistency.py
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
path_to_output = os.path.join(split[0], "inverse_geometric_locomotion/output/")
path_to_output_snake_serpenoid = os.path.join(path_to_output, "snake_spacetime_consistency_timings/")

if not os.path.exists(path_to_output_snake_serpenoid):
    os.makedirs(path_to_output_snake_serpenoid)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import numpy as np
from scipy.optimize import minimize
import time
import torch

from geometry_io import export_snakes_to_json
from objectives import compare_last_translation, grad_compare_last_translation
from snake_shapes import serpenoid_generation_cubic_splines
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from utils import axis_angle_to_quaternion


FLAGS = flags.FLAGS

def obj_and_grad_params(
    params, n_steps, gt, masses, a_weights, b_weights, 
    force_0, torque_0, n_cp, n_points_snake, snake_length, wavelength, close_snake_gait,
    w_fit, fun_anisotropy_dir,
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
        n_points_snake (int): Number of points of the snake.
        snake_length (float): Length of the snake.
        wavelength (float): Wavelength of the snake.
        close_snake_gait (bool): Whether the snake gait is closed or not.
        w_fit (float): Weight of the registration fitting term.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the snake.
        
    Returns:
        obj (float): Objective value.
        grad_params (np.ndarray of shape (n_params,)): Gradient of the objective w.r.t. the parameters.
    '''
    
    params = torch.tensor(params, requires_grad=True)
    example_pos_ = torch.zeros(size=(n_steps, n_points_snake, 3))
    
    pos_ = serpenoid_generation_cubic_splines(
        params[:2*n_cp].reshape(-1, 2), snake_length, wavelength, 
        example_pos_, n_steps, n_points_snake, n_cp, close_snake=close_snake_gait
    )

    pos_np_ = pos_.detach().numpy()
    tangents_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
    
    options = {'maxfev': 2000}
    pos, tangents, g = multiple_steps_forward(
        pos_np_, tangents_np_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )
    
    obj = w_fit * compare_last_translation(g, gt)
    grad_obj_g = w_fit * grad_compare_last_translation(g, gt)

    grad_obj_pos_ = np.zeros_like(pos_np_).reshape(n_steps, -1)
    grad_shape_obj = multiple_steps_backward_pos_(pos_, pos, tangents, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir)
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_steps, n_points_snake, 3)))
    
    grad_obj_params = params.grad.numpy()
    grad_obj_params[:2] = 0.0 # fixing the first CP
    
    return obj, grad_obj_params

def main(_):
    
    list_n_ts = [20, 40, 60]
    list_n_points_snake = [10, 30, 80]
    close_snake_gait = False
    n_cp = 20
    rho = 1.0e-2
    eps = 1.0e-2
    snake_length = 1.0
    wavelength = 1.1
    maxiter = 1000
    
    settings_dict = {
        "n_cp": n_cp,
        "rho": rho,
        "eps": eps,
        "wavelength": wavelength,
        "close_gait": close_snake_gait,
        "maxiter": maxiter,
    }
    
    w_fit = 1.0
    w_fit /= snake_length ** 2
    target_translation = np.array([1.8 * snake_length, 0.9 * snake_length, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).detach().numpy()
    gt = np.concatenate([target_quaternion, target_translation], axis=0)

    def fun_anisotropy_dir(x):
        tangents = torch.zeros_like(x)
        tangents[..., 0, :] = x[..., 1, :] - x[..., 0, :]
        tangents[..., 1:-1, :] = x[..., 2:, :] - x[..., :-2, :]
        tangents[..., -1, :] = x[..., -1, :] - x[..., -2, :]
        tangents = tangents / torch.linalg.norm(tangents, dim=-1, keepdims=True)
        return tangents
    
    center = torch.tensor([0.1, 0.15])
    radius = 2.5
    ts_cp = torch.linspace(0.0, 1.0, n_cp)
    cps = torch.zeros(n_cp, 2)
    cps[:, 0] = center[0] + radius * torch.cos(4 * np.pi * ts_cp[:n_cp+1-close_snake_gait])
    cps[:, 1] = center[1] + radius * torch.sin(4 * np.pi * ts_cp[:n_cp+1-close_snake_gait])

    torch.manual_seed(0)
    cps += 0.4 * torch.randn(n_cp, 2)
    
    for n_ts in list_n_ts:
        for n_points_snake in list_n_points_snake:
            
            settings_dict["n_ts"] = n_ts
            settings_dict["n_pts"] = n_points_snake

            # example_pos_ serves providing the expected shape of pos_
            example_pos_ = torch.zeros(size=(n_ts, n_points_snake, 3))
            pos_ = serpenoid_generation_cubic_splines(
                cps, snake_length, wavelength, 
                example_pos_, n_ts, n_points_snake, n_cp, close_snake=close_snake_gait
            ).numpy()
            tangents_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()

            edge_lengths = np.linalg.norm(pos_[:, 1:] - pos_[:, :-1], axis=-1)
            voronoi_cell_lengths = np.zeros(shape=(n_ts, n_points_snake))
            voronoi_cell_lengths[:, :-1] = edge_lengths / 2.0
            voronoi_cell_lengths[:, 1:] = edge_lengths / 2.0
            masses = rho * np.tile(voronoi_cell_lengths[0], reps=(n_ts, 1))
            a_weight = np.ones(shape=(n_ts, n_points_snake))
            b_weight = (eps - 1.0) * np.ones(shape=(n_ts, n_points_snake))

            force_0 = np.zeros(shape=(3,))
            torque_0 = np.zeros(shape=(3,))
            
            pos, tangents, g = multiple_steps_forward(
                pos_, tangents_, masses, a_weight, b_weight, force_0, torque_0
            )
            
            save_path_init = os.path.join(path_to_output_snake_serpenoid, "snake_init_nts_{}_npts_{}.json".format(n_ts, n_points_snake))

            export_snakes_to_json(
                pos_, g, pos, force_0, torque_0, save_path_init, edges=None,
                weights_optim=None, quantities_per_vertex=None,
                quantities_per_edge=None, target_final_g=gt,
            )
            
            params0 = cps.reshape(-1,).numpy()
            obj_and_grad_params_scipy = lambda x_: obj_and_grad_params(
                x_, n_ts, gt, masses, a_weight, b_weight, force_0, torque_0, n_cp, n_points_snake, snake_length, wavelength, close_snake_gait, w_fit, fun_anisotropy_dir,
            )
            
            start_time = time.time()
            result = minimize(
                obj_and_grad_params_scipy, params0, jac=True, 
                method='L-BFGS-B', options={'disp': True, 'maxiter': maxiter}
            )
            optim_duration = time.time() - start_time
            settings_dict["n_iter"] = result.nit
            
            params_opt = torch.tensor(result.x.copy())
            pos_opt_ = serpenoid_generation_cubic_splines(
                params_opt[:2*n_cp].reshape(-1, 2), snake_length, wavelength, 
                example_pos_, n_ts, n_points_snake, n_cp, close_snake=close_snake_gait
            )
            pos_opt_ = pos_opt_.detach().numpy()
            tangents_opt_ = fun_anisotropy_dir(torch.tensor(pos_opt_)).numpy()

            options = {'maxfev': 2000}
            pos_opt, tangents_opt, g_opt = multiple_steps_forward(
                pos_opt_, tangents_opt_, masses, a_weight, b_weight, force_0, torque_0, options=options
            )

            save_path_opt = os.path.join(path_to_output_snake_serpenoid, "snake_opt_nts_{}_npts_{}.json".format(n_ts, n_points_snake))

            export_snakes_to_json(
                pos_opt_, g_opt, pos_opt, force_0, torque_0, save_path_opt, edges=None,
                weights_optim=None, quantities_per_vertex=None,
                quantities_per_edge=None, target_final_g=gt,
                optimization_duration=optim_duration, optimization_settings=settings_dict,
            )

if __name__ == '__main__':
    app.run(main)