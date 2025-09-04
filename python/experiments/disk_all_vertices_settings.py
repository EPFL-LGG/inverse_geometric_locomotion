import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
path_to_cubic_splines = os.path.join(split[0], "inverse_geometric_locomotion/ext/torchcubicspline/")
_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

import numpy as np
import torch
from utils import axis_angle_to_quaternion

def return_disk_all_vertices_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 80
    setting_dict['rho'] = 1.0
    setting_dict['eps'] = 1.0e-2
    setting_dict['close_gait'] = True
    
    if trial_number in [0]:
        setting_dict['init_perturb_magnitude'] = 1.0e-4
    else:
        setting_dict['init_perturb_magnitude'] = 5.0e-2
    
    if trial_number in [0, 1, 2]:
        setting_dict['max_to_min_weight_ratio'] = 1.0
    elif trial_number in [3]:
        setting_dict['max_to_min_weight_ratio'] = 10.0
    elif trial_number in [4]:
        setting_dict['max_to_min_weight_ratio'] = 75.0

    setting_dict['w_fit'] = 1.0e2
    if trial_number in [0]:
        setting_dict['w_area'], setting_dict['w_edges'], setting_dict['w_bend'] = 0.0, 0.0, 0.0
    elif trial_number in [1]:
        setting_dict['w_area'], setting_dict['w_edges'], setting_dict['w_bend'] = 0.0, 5.0e1, 0.0e-2
    elif trial_number in [2]:
        setting_dict['w_area'], setting_dict['w_edges'], setting_dict['w_bend'] = 0.0, 5.0e1, 5.0e-2
    elif trial_number in [3, 4]:
        setting_dict['w_area'], setting_dict['w_edges'], setting_dict['w_bend'] = 0.0, 5.0e1, 5.0e0
        
    target_translation = np.array([2.0, 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    setting_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    return setting_dict