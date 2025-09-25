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

def return_snake_teaser_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['n_cp'] = 50
    setting_dict['n_ts'] = 200
    setting_dict['rho'] = 1.0e-2
    setting_dict['eps'] = 8.0e-2
    setting_dict['close_gait'] = True
    setting_dict['n_pts'] = 30
    setting_dict['snake_length'] = 1.4
    setting_dict['w_fit'] = 1.0e2
    setting_dict['w_obs'] = 5.0e2
    setting_dict['w_energy'] = 5.0e-1
    setting_dict['init_perturb_magnitude'] = 4.0e-1
    setting_dict['init_wavelength'] = 1.0 / 0.65

    setting_dict['scale_implicit'] = [0.6, 0.6, 0.6]
    setting_dict['translate_implicit'] = [-0.9, -0.2, 0.0]
    setting_dict['angle_rot'], setting_dict['x_obstacle_offset'], setting_dict['x_last_target_offset'] = 1.0 * np.pi / 5.0, 1.0, 1.0
    setting_dict['translation_siggraph'] = [- setting_dict['translate_implicit'][0] + setting_dict['scale_implicit'][0] + setting_dict['x_obstacle_offset'], 0.0, 0.0]
        
    target_translation1 = np.array([
        setting_dict['scale_implicit'][0] + 1.75 * setting_dict['x_obstacle_offset'], 
        1.0 * setting_dict['scale_implicit'][1], 0.0
    ])
    target_translation2 = np.array([
        setting_dict['translation_siggraph'][0], 
        -1.0 * setting_dict['scale_implicit'][1], 0.0
    ])
    target_translation4 = np.array([
        setting_dict['translation_siggraph'][0] - setting_dict['translate_implicit'][0] + setting_dict['x_last_target_offset'], 
        -0.1 * setting_dict['scale_implicit'][1], 0.0
    ])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).detach().numpy()
    gt = np.stack([
        np.concatenate([target_quaternion, target_translation1], axis=0),
        np.concatenate([target_quaternion, target_translation2], axis=0),
        np.concatenate([target_quaternion, target_translation4], axis=0),
    ], axis=0)
    setting_dict['gt'] = gt.tolist()
    
    setting_dict['min_wavelength'] = 1.0 / 1.5
    setting_dict['max_wavelength'] = 1.0 / 0.5
    
    return setting_dict