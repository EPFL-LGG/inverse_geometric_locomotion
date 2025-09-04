import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
_sys.path.insert(0, path_to_python_scripts)

import json
import numpy as np
import torch
from utils import axis_angle_to_quaternion

def return_sierpinsky_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['n_ts'] = 100
    setting_dict['n_cp'] = 20
    setting_dict['n_pts'] = -1
    setting_dict['rho'] = 1.0
    setting_dict['eps'] = 1.0e-2
    setting_dict['close_gait'] = True
    setting_dict['w_fit'] = 1.0
    setting_dict['w_energy'] = 0.0
    setting_dict['init_perturb_magnitude'] = 1.0e-1

    target_translation = np.array([0.0, 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    if trial_number in [0]:
        target_translation = np.array([1.0, 0.0, 0.0])
    elif trial_number in [1]:
        target_rotation_axis = torch.tensor([0.0, 0.0, torch.pi])
    elif trial_number in [2]:
        target_translation = np.array([0.01, 0.0, 0.5])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).detach().numpy()
    gt = np.concatenate([target_quaternion, target_translation], axis=0)
    setting_dict['gt'] = gt.tolist()

    return setting_dict