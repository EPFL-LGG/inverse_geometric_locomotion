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

def return_snake_straight_traj_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 500
    setting_dict['n_ts'] = 100
    setting_dict['n_cp'] = 20
    setting_dict['n_pts'] = 80
    setting_dict['rho'] = 1.0e-2
    setting_dict['eps'] = 1.0e-1
    setting_dict['close_gait'] = True
    setting_dict['snake_length'] = 1.0
    setting_dict['w_fit'] = 1.0e1
    setting_dict['w_energy'] = [0.0e1, 1.0e-2][trial_number]
    setting_dict['init_wavelength'] = 1.0 / 0.9

    target_translation = np.array([1.4 * setting_dict['snake_length'], 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).detach().numpy()
    gt = np.concatenate([target_quaternion, target_translation], axis=0)
    setting_dict['gt'] = gt.tolist()

    setting_dict['min_wavelength'] = 1.0 / 1.5
    setting_dict['max_wavelength'] = 1.0 / 0.5

    center = torch.tensor([0.1, 0.15])
    radius = 3.0
    cycles = 2.0
    ts_cp = torch.linspace(0.0, 1.0, setting_dict['n_cp'])
    cps = torch.zeros(setting_dict['n_cp'], 3)
    cps[:, 0] = center[0] + radius * torch.cos(cycles * 2.0 * np.pi * ts_cp)
    cps[:, 1] = center[1] + radius * torch.sin(cycles * 2.0 * np.pi * ts_cp)
    cps[:, 2] = setting_dict['init_wavelength']
    setting_dict["cps"] = cps.tolist()

    return setting_dict