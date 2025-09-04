import json
import numpy as np
import torch

def export_snakes_to_json(
    pos_, g, pos, force_0, torque_0, save_path, edges=None,
    weights_optim=None, quantities_per_vertex=None,
    quantities_per_edge=None, target_final_g=None,
    target_checkpoints_g=None, obstacle=None, 
    optimization_settings=None, optimization_duration=None,
    optimization_evolution=None, mass_data=None,
):
    '''
    Args:
        pos_: torch.tensor/np.array of shape (n_ts, n_points 3)
        g: torch.tensor/np.array of shape (n_ts, 7)
        pos: torch.tensor/np.array of shape (n_ts, n_points 3)
        force_0: torch.tensor/np.array of shape (3,)
        torque_0: torch.tensor/np.array of shape (3,)
        edges: torch.tensor/np.array of shape (M, 2)
        save_path: str
        weights_optim: dict of optimization weights
        quantities_per_vertex: dict of lists of shapes (n_ts, N)
        quantities_per_edge: dict of lists of shapes (n_ts, M)
        target_final_g: torch.tensor/np.array of shape (7,)
        target_checkpoints_g: torch.tensor/np.array of shape (n_checkpoints, 7)
        obstacle: an object of type ImplicitFunction
        optimization_settings: dict of optimization settings
        optimization_duration: float, the duration of the optimization
        optimization_evolution: dict of optimization evolution
        mass_data: dict of masses, may contain logits/min/max masses
    '''
    if edges is None:
        edges = torch.stack([
            torch.arange(0, pos.shape[1]-1, dtype=torch.long),
            torch.arange(1, pos.shape[1], dtype=torch.long),
        ], dim=1)
    
    if weights_optim is None:
        weights_optim = {}
        
    if quantities_per_vertex is None:
        quantities_per_vertex = {}
        
    if quantities_per_edge is None:
        quantities_per_edge = {}
        
    if target_final_g is None:
        target_final_g = torch.tensor([])
        
    if target_checkpoints_g is None:
        target_checkpoints_g = torch.tensor([])
        
    if obstacle is None:
        obstacle_ser = None
    else:
        obstacle_ser = obstacle.serialize()
        
    if optimization_settings is None:
        optimization_settings = {}
        
    if optimization_duration is None:
        optimization_duration = -1.0
        
    if optimization_evolution is None:
        optimization_evolution = {}

    if mass_data is None:
        mass_data = {}

    with open(save_path, 'w') as jsonFile:
        json.dump({
            'pos_': pos_.tolist(),
            'g': g.tolist(),
            'pos': pos.tolist(),
            'force_0': force_0.tolist(),
            'torque_0': torque_0.tolist(),
            'edges': edges.tolist(),
            'weights_optim': weights_optim,
            'quantities_per_vertex': quantities_per_vertex,
            'quantities_per_edge': quantities_per_edge,
            'target_final_g': target_final_g.tolist(),
            'target_checkpoints_g': target_checkpoints_g.tolist(),
            'obstacle': obstacle_ser,
            'optimization_settings': optimization_settings,
            'optimization_duration': optimization_duration,
            'optimization_evolution': optimization_evolution,
            'mass_data': mass_data,
        }, jsonFile, indent=4)

def export_meshes_to_json(
    pos_, g, pos, force_0, torque_0, edges, faces, save_path, 
    weights_optim=None, quantities_per_vertex=None,
    quantities_per_edge=None,
    quantities_per_face=None, target_final_g=None,
    target_checkpoints_g=None, obstacle=None,
    handle_ids=None,
    optimization_settings=None, optimization_duration=None,
    optimization_evolution=None, mass_data=None,
):
    '''
    Args:
        pos_: torch.tensor/np.array of shape (n_ts, n_points 3)
        g: torch.tensor/np.array of shape (n_ts, 7)
        pos: torch.tensor/np.array of shape (n_ts, n_points 3)
        force_0: torch.tensor/np.array of shape (3,)
        torque_0: torch.tensor/np.array of shape (3,)
        edges: torch.tensor/np.array of shape (n_edges, 2)
        faces: torch.tensor/np.array of shape (M, 3)
        save_path: str
        weights_optim: dict of optimization weights
        quantities_per_vertex: dict of lists of shapes (n_ts, N)
        quantities_per_edge: dict of lists of shapes (n_ts, n_edges)
        quantities_per_face: dict of lists of shapes (n_ts, M)
        target_final_g: torch.tensor/np.array of shape (7,)
        target_checkpoints_g: torch.tensor/np.array of shape (n_checkpoints, 7)
        obstacle: an object of type ImplicitFunction
        handle_ids: list of ints representing the indices of the handles
        optimization_settings: dict of optimization settings
        optimization_duration: float, the duration of the optimization
        optimization_evolution: dict of optimization evolution
    '''
    if weights_optim is None:
        weights_optim = {}
        
    if quantities_per_vertex is None:
        quantities_per_vertex = {}
        
    if quantities_per_edge is None:
        quantities_per_edge = {}
        
    if quantities_per_face is None:
        quantities_per_face = {}
        
    if target_final_g is None:
        target_final_g = torch.tensor([])
        
    if target_checkpoints_g is None:
        target_checkpoints_g = torch.tensor([])
        
    if obstacle is None:
        obstacle_ser = None
    else:
        obstacle_ser = obstacle.serialize()
        
    if handle_ids is None:
        handle_ids = []
        
    if optimization_settings is None:
        optimization_settings = {}

    if optimization_duration is None:
        optimization_duration = -1.0
        
    if optimization_evolution is None:
        optimization_evolution = {}

    if mass_data is None:
        mass_data = {}

    with open(save_path, 'w') as jsonFile:
        json.dump({
            'pos_': pos_.tolist(),
            'g': g.tolist(),
            'pos': pos.tolist(),
            'force_0': force_0.tolist(),
            'torque_0': torque_0.tolist(),
            'edges': edges.tolist(),
            'faces': faces.tolist(),
            'weights_optim': weights_optim,
            'quantities_per_vertex': quantities_per_vertex,
            'quantities_per_edge': quantities_per_edge,
            'quantities_per_face': quantities_per_face,
            'target_final_g': target_final_g.tolist(),
            'target_checkpoints_g': target_checkpoints_g.tolist(),
            'obstacle': obstacle_ser,
            'handle_ids': handle_ids,
            'optimization_settings': optimization_settings,
            'optimization_duration': optimization_duration,
            'optimization_evolution': optimization_evolution,
            'mass_data': mass_data,
        }, jsonFile, indent=4)
        
def export_graphs_to_json(
    pos_, g, pos, force_0, torque_0, edges, save_path, 
    weights_optim=None, quantities_per_vertex=None,
    quantities_per_edge=None, target_final_g=None,
    target_checkpoints_g=None, obstacle=None,
    optimization_settings=None, optimization_duration=None,
    optimization_evolution=None, mass_data=None,
):
    '''
    Args:
        pos_: torch.tensor/np.array of shape (n_ts, n_points 3)
        g: torch.tensor/np.array of shape (n_ts, 7)
        pos: torch.tensor/np.array of shape (n_ts, n_points 3)
        force_0: torch.tensor/np.array of shape (3,)
        torque_0: torch.tensor/np.array of shape (3,)
        edges: torch.tensor/np.array of shape (M, 2)
        save_path: str
        weights_optim: dict of optimization weights
        quantities_per_vertex: dict of lists of shapes (n_ts, N)
        quantities_per_edge: dict of lists of shapes (n_ts, M)
        target_final_g: torch.tensor/np.array of shape (7,)
        target_checkpoints_g: torch.tensor/np.array of shape (n_checkpoints, 7)
        obstacle: an object of type ImplicitFunction
        optimization_settings: dict of optimization settings
        optimization_duration: float, the duration of the optimization
        optimization_evolution: dict of optimization evolution
    '''
    if weights_optim is None:
        weights_optim = {}
        
    if quantities_per_vertex is None:
        quantities_per_vertex = {}
        
    if quantities_per_edge is None:
        quantities_per_edge = {}
        
    if target_final_g is None:
        target_final_g = torch.tensor([])
        
    if target_checkpoints_g is None:
        target_checkpoints_g = torch.tensor([])
        
    if obstacle is None:
        obstacle_ser = None
    else:
        obstacle_ser = obstacle.serialize()
        
    if optimization_settings is None:
        optimization_settings = {}

    if optimization_duration is None:
        optimization_duration = -1.0
        
    if optimization_evolution is None:
        optimization_evolution = {}

    if mass_data is None:
        mass_data = {}

    with open(save_path, 'w') as jsonFile:
        json.dump({
            'pos_': pos_.tolist(),
            'g': g.tolist(),
            'pos': pos.tolist(),
            'force_0': force_0.tolist(),
            'torque_0': torque_0.tolist(),
            'edges': edges.tolist(),
            'weights_optim': weights_optim,
            'quantities_per_vertex': quantities_per_vertex,
            'quantities_per_edge': quantities_per_edge,
            'target_final_g': target_final_g.tolist(),
            'target_checkpoints_g': target_checkpoints_g.tolist(),
            'obstacle': obstacle_ser,
            'optimization_settings': optimization_settings,
            'optimization_duration': optimization_duration,
            'optimization_evolution': optimization_evolution,
            'mass_data': mass_data,
        }, jsonFile, indent=4)
        
def read_mesh_obj_triangles_quads(file_name):
    """Read an OBJ mesh file, the specificity comes from colours being embedded as vertices.

    Parameters
    ----------
    file_name : str
        The path of the OBJ file to open.

    Returns
    -------
    v : np.array (V,3)
        The array of vertices [[x_0, y_0, z_0], ..., [x_V, y_V, z_V]].
    f : list
        The list of faces [[v_i, v_j, ...], ...]

    Notes
    -----
    Files must be written without line wrap (in many CAD software,
    line wrap can be disabled in the OBJ saving options).
    """
    if not file_name.endswith('.obj'):
        file_name = file_name + '.obj'
    file_name = str(file_name)
    obj_file = open(file_name, encoding='utf-8')
    vertices_list = []
    f = []
    for line in obj_file:
        split_line = line.split(' ')
        split_line = [elem for elem in split_line if elem != '']
        if split_line[0] == 'v':
            split_x = split_line[1].split('\n')
            x = float(split_x[0])
            split_y = split_line[2].split('\n')
            y = float(split_y[0])
            split_z = split_line[3].split('\n')
            try:
                z = float(split_z[0])
            except ValueError:
                print('WARNING: disable line wrap when saving .obj')
            vertices_list.append([x, y, z])
        elif split_line[0] == 'f':
            v_list = []
            L = len(split_line)
            try:
                for i in range(1, L - 1):
                    split_face_data = split_line[i].split('/')
                    v_list.append(int(split_face_data[0]) - 1)
                f.append(v_list)
            except ValueError:
                v_list = []
                for i in range(1, L - 1):
                    v_list.append(int(split_line[i]) - 1)
                f.append(v_list)
    v = np.array(vertices_list)
    f_triangles = [face for face in f if len(face) == 3]
    f_quads = [face for face in f if len(face) == 4]
    try:
        f_triangles = np.array(f_triangles)
        f_quads = np.array(f_quads)
    except TypeError:
        pass
    return v, f_triangles, f_quads

def read_mesh_obj_lines(file_name):
    """Read an OBJ mesh file, the specificity comes from colours being embedded as vertices.

    Parameters
    ----------
    file_name : str
        The path of the OBJ file to open.

    Returns
    -------
    v : np.array (V, 3)
        The array of vertices [[x_0, y_0, z_0], ..., [x_V, y_V, z_V]].
    lines_list : list
        The list of lines [[v_i, v_j, ...], ...]

    Notes
    -----
    Files must be written without line wrap (in many CAD software,
    line wrap can be disabled in the OBJ saving options).
    """
    if not file_name.endswith('.obj'):
        file_name = file_name + '.obj'
    file_name = str(file_name)
    obj_file = open(file_name, encoding='utf-8')
    vertices_list = []
    lines_list = []
    for line in obj_file:
        split_line = line.split(' ')
        split_line = [elem for elem in split_line if elem != '']
        if split_line[0] == 'v':
            split_x = split_line[1].split('\n')
            x = float(split_x[0])
            split_y = split_line[2].split('\n')
            y = float(split_y[0])
            split_z = split_line[3].split('\n')
            try:
                z = float(split_z[0])
            except ValueError:
                print('WARNING: disable line wrap when saving .obj')
            vertices_list.append([x, y, z])
        elif split_line[0] == 'l':
            v_list = []
            L = len(split_line)
            try:
                for i in range(1, L - 1):
                    split_face_data = split_line[i].split('/')
                    v_list.append(int(split_face_data[0]) - 1)
                lines_list.append(v_list)
            except ValueError:
                v_list = []
                for i in range(1, L - 1):
                    v_list.append(int(split_line[i]) - 1)
                lines_list.append(v_list)
    v = np.array(vertices_list)
    return v, lines_list
