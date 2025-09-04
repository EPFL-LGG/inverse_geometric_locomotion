import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
path_to_output = os.path.join(split[0], "inverse_geometric_locomotion/output/")
path_to_data = os.path.join(path_to_output, "snake_confined/")
_sys.path.insert(0, path_to_python_scripts)

import json
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
    setting_dict['maxiter'] = 10
    setting_dict['n_ts'] = 100
    setting_dict['n_cp'] = 20
    setting_dict['n_pts'] = 80
    setting_dict['rho'] = 1.0e-2
    setting_dict['eps'] = 1.0e-1
    setting_dict['close_gait'] = True
    setting_dict['snake_length'] = 1.0
    setting_dict['w_fit'] = 1.0
    setting_dict['w_energy'] = 0.0e1
    setting_dict['init_wavelength'] = 1.0 / 0.9
    
    target_translation = np.array([0.0, 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, torch.pi - 1.0e-3])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).detach().numpy()
    gt = np.concatenate([target_quaternion, target_translation], axis=0)
    setting_dict['gt'] = gt.tolist()

    if trial_number in [0, 1, 2]:
        setting_dict['boxsize'] = 0.5 - 1.0e-2
    elif trial_number in [3]:
        setting_dict['boxsize'] = 0.5 - 1.0e-1

    if trial_number in [0, 1]:
        setting_dict['w_obs'] = 0.0e1
    elif trial_number in [2, 3]:
        setting_dict['w_obs'] = 1.0e2
        
    if trial_number in [0, 2, 3]:
        setting_dict['w_com'] = 0.0e1
    elif trial_number in [1]:
        setting_dict['w_com'] = 2.0e0
    
    setting_dict['alpha_rep'] = 3.0
    setting_dict['beta_rep'] = 6.0
    setting_dict['n_edge_disc'] = 1

    if trial_number in [0]:
        setting_dict['w_rep'] = 1.0e0
    elif trial_number in [1, 2, 3]:
        setting_dict['w_rep'] = 3.0e0

    setting_dict['w_ecr'] = 0.0
    
    mult_length_threshold = 5.0
    setting_dict['length_threshold'] = mult_length_threshold * setting_dict['snake_length'] / (setting_dict['n_pts'] - 1.0) / (max(1, setting_dict['n_edge_disc']))

    setting_dict['min_wavelength'] = 1.0 / 1.5
    setting_dict['max_wavelength'] = 1.0 / 0.5

    print(setting_dict['length_threshold'])
    
    if trial_number in [0]:
        center = torch.tensor([0.1, 0.15])
        radius = 3.0
        cycles = 2.0
        ts_cp = torch.linspace(0.0, 1.0, setting_dict['n_cp'])
        cps = torch.zeros(setting_dict['n_cp'], 3)
        cps[:, 0] = center[0] + radius * torch.cos(cycles * 2.0 * np.pi * ts_cp)
        cps[:, 1] = center[1] + radius * torch.sin(cycles * 2.0 * np.pi * ts_cp)
        cps[:, 2] = setting_dict['init_wavelength']
        cps[0, 0] = 5.0
        cps[0, 1] = 6.0
        setting_dict["cps"] = cps.tolist()
    elif trial_number in [1]:
        exp_fn = "snake_confined_opt_00.json"
        with open(os.path.join(path_to_data, exp_fn), encoding='utf-8') as json_file:
            js_load = json.load(json_file)
        setting_dict["cps"] = np.array(js_load['optimization_settings']["params_opt"]).reshape(-1, 3).tolist()
        print(setting_dict["cps"][0])
    elif trial_number in [2, 3]:
        exp_fn = "snake_confined_opt_01.json"
        with open(os.path.join(path_to_data, exp_fn), encoding='utf-8') as json_file:
            js_load = json.load(json_file)
        setting_dict["cps"] = np.array(js_load['optimization_settings']["params_opt"]).reshape(-1, 3).tolist()
        print(setting_dict["cps"][0])

    return setting_dict