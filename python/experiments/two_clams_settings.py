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

def return_two_clams_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['maxiter'] = 10
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 200
    setting_dict['rho'] = 1.0
    setting_dict['eps'] = 0.5
    setting_dict['close_gait'] = True
    setting_dict['n_pts_per_segment'] = 30
    setting_dict['n_points_clams'] = 6 + 5 * setting_dict['n_pts_per_segment']
    setting_dict['w_fit'] = 1.0e1
    setting_dict['w_energy'] = 0.0e1
    setting_dict['init_perturb_magnitude'] = 0.0e-2

    setting_dict['min_abs_angles'], setting_dict['max_abs_angles'] = 0.1, 1.0 * np.pi

    setting_dict['total_length'] = 1.0
    setting_dict['eta'] = 0.747
    setting_dict['e1bye2'], setting_dict['e2bye3'] = 1.0 / setting_dict['eta'], 1.0 / setting_dict['eta']

    setting_dict['target_disp_norm'] = 0.25
    setting_dict['target_disp_direction'] = None

    return setting_dict