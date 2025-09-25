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

def return_snake_ff_obstacle_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 100
    setting_dict['rho'] = 1.0e-2
    setting_dict['n_ts'] = 200
    setting_dict['eps'] = 5.0e-2
    setting_dict['close_gait'] = True
    setting_dict['n_pts'] = 11
    setting_dict['n_angles'] = setting_dict['n_pts'] - 2
    setting_dict['snake_length'] = 1.0
    setting_dict['init_perturb_magnitude'] = 1.0e-3

    setting_dict['min_turning_angle'], setting_dict['max_turning_angle'] = - 0.35 * np.pi, 0.35 * np.pi

    setting_dict['broken_joint_ids'] = []
    setting_dict['broken_joint_angles'] = []

    setting_dict['obstacle_params'] = [
        1.25 * setting_dict['snake_length'], 0.0, 0.0, 
        0.5 * setting_dict['snake_length']
    ]
    target_translation = np.array([2.5 * setting_dict['snake_length'], 0.0, 0.0])

    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).detach().numpy()
    gt = np.concatenate([target_quaternion, target_translation], axis=0)
    setting_dict['gt'] = gt.tolist()
    
    setting_dict['w_fit'] = 1.0e2
    setting_dict['w_obs'] = [0.0, 1.0e2][trial_number]
    setting_dict['w_energy'] = 1.0e1

    return setting_dict