import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_geolab = os.path.join(split[0], "inverse_geometric_locomotion/ext/geolab/")
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
path_to_data_cat = os.path.join(split[0], "inverse_geometric_locomotion/data/Cat_rig_obj_cache/")
_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_geolab)

from geolab import read_mesh
import glob
import numpy as np
import pypandoc
import re
import torch
from utils import axis_angle_to_quaternion, quaternion_to_matrix_torch, nostdout

def return_cat_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    setting_dict = {}
    
    setting_dict['maxiter'] = 1000
    setting_dict['n_cp'] = 15
    setting_dict['n_ts'] = 100
    setting_dict['rho'] = 1.0
    setting_dict['eps'] = 1.0
    setting_dict['close_gait'] = True
    setting_dict['w_fit'] = 1.0e2
    setting_dict['w_edges'] = 1.0e2
    setting_dict['init_perturb_magnitude'] = 1.0e-2
    
    setting_dict['w_gait_fit'] = [0.0e1, 1.0e-1][trial_number]
    
    target_translation = np.array([0.0, 0.0, 0.0])
    target_rotation_axis = torch.tensor([-torch.pi, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    setting_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()
    
    n_pts, vertices, edges_graph, initial_edge_lengths, vertices_ref, vertices_perturbed, pos_files_ = get_cat_initial_gait_from_experiment(trial_number)
    setting_dict['n_pts'] = n_pts
    setting_dict['vertices'] = vertices.tolist()
    setting_dict['edges_graph'] = edges_graph.tolist()
    setting_dict['initial_edge_lengths'] = initial_edge_lengths.tolist()
    setting_dict['vertices_ref'] = vertices_ref.tolist()
    setting_dict['vertices_perturbed'] = vertices_perturbed.tolist()
    setting_dict['pos_files_'] = pos_files_.tolist()
    
    alpha_interp = torch.linspace(0.0, 1.0, setting_dict['n_cp']).reshape(-1, 1)
    cps_interp = (1.0 - alpha_interp) * torch.tensor(setting_dict['pos_files_'])[0].reshape(1, -1) + alpha_interp * torch.tensor(setting_dict['pos_files_'])[-1].reshape(1, -1)
    cps = cps_interp - torch.tensor(setting_dict['vertices']).reshape(1, -1)
    torch.manual_seed(0)
    cps[1:-1] += setting_dict['init_perturb_magnitude'] * torch.randn_like(cps[1:-1])
    setting_dict['cps'] = cps.tolist()

    return setting_dict

def get_cat_initial_gait_from_file():
    pattern = os.path.join(path_to_data_cat, "Cat_rig_obj_cache_*.obj")
    matching_files = glob.glob(pattern)
    n_files = len(matching_files)

    pattern_re = os.path.join(path_to_data_cat, r"Cat_rig_obj_cache_(\d+)\.obj")
    def extract_integer(filename):
        match = re.search(pattern_re, filename)
        return int(match.group(1)) if match else -1

    sorted_files = sorted(matching_files, key=extract_integer)

    with nostdout():
        for id_file, fn in enumerate(sorted_files):
            if id_file == 0:
                vertices, _ = read_mesh(fn)
                pos_files_ = np.zeros(shape=(n_files, vertices.shape[0], 3))
                pos_files_[id_file] = vertices
            pos_files_[id_file], _ = read_mesh(fn)
        
    n_points = vertices.shape[0]

    # Align the cat so that it starts paws up and its body is along the x-axis
    angle = 2.0 * np.pi / 3.0
    axis = torch.tensor([1.0, 1.0, 1.0])
    axis /= torch.linalg.norm(axis)
    rot_mat = quaternion_to_matrix_torch(axis_angle_to_quaternion(angle * axis))
    vertices = vertices @ rot_mat.numpy().T
    pos_files_ = pos_files_ @ rot_mat.numpy().T

    edges_file = os.path.join(path_to_data_cat, "edge_adjacency_cat.rtf")
    text = pypandoc.convert_file(edges_file, 'plain')
    edges_graph = []
    for line in text.split("\n"):
        if len(line)>3:
            edges_graph.append([int(x) for x in line.split("[")[1].split("]")[0].split(", ")])
    edges_graph = np.array(edges_graph)
    initial_edge_lengths = np.linalg.norm(vertices[edges_graph[:, 0]] - vertices[edges_graph[:, 1]], axis=-1)
    vertices_ref = vertices.copy()
    torch.manual_seed(0)
    vertices_perturbed = vertices + 1.0e-3 * torch.randn_like(torch.tensor(vertices)).numpy()
    
    return n_points, vertices, edges_graph, initial_edge_lengths, vertices_ref, vertices_perturbed, pos_files_

def get_cat_initial_gait_from_experiment(trial_number):
    '''Can be useful when the initial gait changes with the trial number'''
    return get_cat_initial_gait_from_file()