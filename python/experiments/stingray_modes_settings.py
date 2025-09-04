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

def return_stingray_modes_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['w_fit'] = 1.0e2
    setting_dict['maxiter'] = 1000
    setting_dict['maxiter'] = 10
    setting_dict['n_cp'] = 15
    setting_dict['rho'] = 1.0
    setting_dict['close_gait'] = True
    setting_dict['eps'] = 1.0e-4    
    setting_dict['n_ts'] = 100
    setting_dict['init_perturb_magnitude'] = 5.0e-3    
    setting_dict['n_modes'] = 40
    setting_dict['max_to_min_weight_ratio'] = 25.0

    setting_dict['w_fit_cp'], setting_dict['w_vol'], setting_dict['w_edges'], setting_dict['w_bend'] = 0.0e1, 1.0e1, 5.0e1, 5.0e-2
    setting_dict['w_g_speed'] = 1.0e1
    
    angle_target_disp = - 0.05 * np.pi
    rad_target_disp = 3.3
    target_translation = rad_target_disp * np.array([np.cos(angle_target_disp), np.sin(angle_target_disp), 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, angle_target_disp])
    
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    gt = np.concatenate([target_quaternion, target_translation], axis=0)
    setting_dict['gt'] = gt.tolist()
    setting_dict['gcp'] = gt.tolist()

    return setting_dict