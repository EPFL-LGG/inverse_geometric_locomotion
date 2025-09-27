import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
_sys.path.insert(0, path_to_python_scripts)

import numpy as np
import torch
from utils import axis_angle_to_quaternion

def return_turtle_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 100
    setting_dict['rho'] = 1.0
    setting_dict['eps'] = 1.0e-2
    setting_dict['close_gait'] = True
    setting_dict['n_pts_per_segment'] = 20
    n_points_turtle = 10 + 9 * setting_dict['n_pts_per_segment']
    setting_dict['w_fit'], setting_dict['w_bound'] = 1.0e1, 1.0e1
    setting_dict['init_perturb_magnitude'] = 2.0e-2
    
    setting_dict['max_alpha_front'], setting_dict['max_alpha_back'] = 0.45 * np.pi, 0.45 * np.pi
    setting_dict['min_alpha_front'], setting_dict['min_alpha_back'] = 0.08 * np.pi, 0.05 * np.pi
    
    setting_dict['front_beta'] = np.pi / 10.0
    setting_dict['back_beta'] = 0.0

    setting_dict['l_trunk'], setting_dict['l_front_arm'], setting_dict['l_front_forearm'], setting_dict['l_back_arm'], setting_dict['l_back_forearm'] = 1.0, 0.3, 0.8, 0.4, 0.4

    rho = 1.0 * np.ones(shape=(1, n_points_turtle))
    rho[0, :setting_dict['n_pts_per_segment']] = 4.0
    setting_dict['rho'] = rho.tolist()

    setting_dict['w_com'], setting_dict['w_energy'] = 0.0e0, 2.0e-2

    angle_target_disp = 0.95 * np.pi
    rad_target_disp = 0.6
    target_translation = rad_target_disp * np.array([np.cos(angle_target_disp), np.sin(angle_target_disp), 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, torch.pi - angle_target_disp])

    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    setting_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    return setting_dict