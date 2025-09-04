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

def return_snake_broken_joint_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['maxiter'] = 10
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 100
    setting_dict['rho'] = 1.0e-2
    setting_dict['eps'] = 1.0e-1
    setting_dict['close_gait'] = True
    setting_dict['n_pts'] = 11
    setting_dict['n_angles'] = setting_dict['n_pts'] - 2
    setting_dict['snake_length'] = 1.0
    setting_dict['w_fit'] = 1.0e2
    setting_dict['w_bound'] = 1.0e1
    setting_dict['init_perturb_magnitude'] = 1.0e-2
    
    setting_dict['min_turning_angle'], setting_dict['max_turning_angle'] = - 0.35 * np.pi, 0.35 * np.pi
    
    if trial_number in [0, 1]:
        setting_dict['broken_joint_ids'] = []
        setting_dict['broken_joint_angles'] = []
    elif trial_number in [2, 3]:
        setting_dict['broken_joint_ids'] = [int((setting_dict['n_angles']) // 2)]
        setting_dict['broken_joint_angles'] = [np.pi / 4.0]

    if trial_number in [0, 2]:
        setting_dict['w_energy'] = 0.0e1
    elif trial_number in [1, 3]:
        setting_dict['w_energy'] = 1.0e1

    target_translation = np.array([1.3 * setting_dict['snake_length'], 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    setting_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    return setting_dict