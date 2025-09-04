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

def return_sphere_handles_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 200
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 80
    setting_dict['rho'] = 1.0
    setting_dict['eps'] = 1.0e-2
    setting_dict['close_gait'] = True
    setting_dict['init_perturb_magnitude'] = 5.0e-2

    setting_dict['w_fit'] = 1.0e2
    if trial_number == 0:
        setting_dict['w_vol'], setting_dict['w_edges'] = 0.0e1, 0.0e1
    elif trial_number == 1:
        setting_dict['w_vol'], setting_dict['w_edges'] = 1.0e1, 0.0e1
    elif trial_number == 2:
        setting_dict['w_vol'], setting_dict['w_edges'] = 0.0e1, 1.0e1
    elif trial_number == 3:
        setting_dict['w_vol'], setting_dict['w_edges'] = 1.0e1, 1.0e1
        
    target_translation = np.array([0.6, 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    setting_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    return setting_dict