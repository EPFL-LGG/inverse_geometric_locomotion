"""This script can be executed to let a triangular robot flap its wings.

In a terminal, with the conda environment turned on, run the following command line:

python sierpinsky.py --trial_number=0
"""

import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("inverse_geometric_locomotion")
if len(split)<2:
    print("Please rename the repository 'inverse_geometric_locomotion'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "inverse_geometric_locomotion/python/")
path_to_cubic_splines = os.path.join(split[0], "inverse_geometric_locomotion/ext/torchcubicspline/")
path_to_notifications = os.path.join(split[0], "inverse_geometric_locomotion/notebooks/notifications/")
path_to_output = os.path.join(split[0], "inverse_geometric_locomotion/output/")
path_to_output_sierpinsky = os.path.join(path_to_output, "sierpinsky/")

if not os.path.exists(path_to_output_sierpinsky):
    os.makedirs(path_to_output_sierpinsky)

_sys.path.insert(0, path_to_python_scripts)
_sys.path.insert(0, path_to_cubic_splines)

from absl import app
from absl import flags
import igl
import numpy as np
from scipy.optimize import minimize, Bounds
import torch
import time

from geometry_io import export_meshes_to_json
from objectives import compare_last_registration, grad_compare_last_registration
from objectives import energy_path, grad_energy_path
from scipy.optimize import minimize, Bounds
from sierpinsky_settings import return_sierpinsky_experiment_settings
from sierpinsky_shapes import sierpinski_deformation_from_cubic_spline, sierpinski_triangle_dense
from step_backward import multiple_steps_backward_pos_
from step_forward import multiple_steps_forward
from surface_energy import compute_mesh_vertex_normals, vmap_compute_mesh_vertex_normals
from utils import print_quaternion

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

FLAGS = flags.FLAGS
flags.DEFINE_integer("trial_number", 0, "The trial number for the experiment.")

def obj_and_grad_params(
    params, n_steps, gt, masses, a_weights, b_weights, 
    force_0, torque_0, n_cp, close_gait,
    vertices_ref, vertex_face_ids, hinge_vertex_ids,
    w_fit_scaled, w_energy_scaled,
    fun_anisotropy_dir, fun_obj_grad_g,
):
    '''
    Args:
        params (np.ndarray of shape (n_params,)): Parameters of the optimization problem.
        n_steps (int): Number of steps of the optimization problem.
        gt (np.ndarray of shape (7,)): Target orientation.
        masses (np.ndarray of shape (n_steps, n_points)): Masses of the object.
        a_weights (np.ndarray of shape (n_steps, n_points)): Weights of the object.
        b_weights (np.ndarray of shape (n_steps, n_points)): Weights of the object.
        force_0 (np.ndarray of shape (3,)): External force applied to the object.
        torque_0 (np.ndarray of shape (3,)): External torque applied to the object.
        n_cp (int): Number of control points of the object.
        close_gait (bool): Whether the object gait is closed or not.
        vertices_ref (torch.tensor of shape (n_vertices, 3)): The vertices of the sierpinsky triangle
        vertex_face_ids (torch.tensor of shape (n_vertices,)): The id of the big triangle each vertex belongs to (3 for the center vertices)
        hinge_vertex_ids (torch.tensor of shape (3, 2)): The vertices forming each hinge in the triangle
        w_*_scaled (float): The weight of the corresponding term.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the object.
        fun_obj_grad_g (callable): Function that computes the objective and its gradient w.r.t. the orientation.

    Returns:
        obj (float): Objective value.
        grad_params (np.ndarray of shape (n_params,)): Gradient of the objective w.r.t. the parameters.
    '''
    
    params = torch.tensor(params, requires_grad=True)

    cps_torch = params.reshape(n_cp, -1)
    pos_ = sierpinski_deformation_from_cubic_spline(
        cps_torch, torch.tensor(vertices_ref), torch.tensor(vertex_face_ids), 
        torch.tensor(hinge_vertex_ids), n_steps, n_cp, close_gait=close_gait
    )

    pos_np_ = pos_.detach().numpy()
    normals_np_ = fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
    
    options = {'maxfev': 2000}
    pos, normals, g = multiple_steps_forward(
        pos_np_, normals_np_, masses, a_weights, b_weights, force_0, torque_0, options=options,
    )
    
    obj, grad_obj_g, grad_obj_pos_ = fun_obj_grad_g(g, pos_np_, gt, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled,  fun_anisotropy_dir)
    grad_obj_pos_ = grad_obj_pos_.reshape(n_steps, -1)
    grad_shape_obj = multiple_steps_backward_pos_(pos_np_, pos, normals, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir)
    pos_.backward(torch.tensor(grad_shape_obj.reshape(n_steps, -1, 3)))

    grad_params = params.grad.numpy()
    
    return obj, grad_params

def fun_obj_grad_g(gs, pos_, gt, masses, a_weights, b_weights, w_fit_scaled, w_energy_scaled, fun_anisotropy_dir):
    '''
    Args:
        gs: torch.tensor of shape (n_ts, 7)
        pos_: torch.tensor of shape (n_ts, n_points, 3)
        gt: np.ndarray of shape (7,)
        masses: np.ndarray of shape (n_ts, n_points)
        a_weights: np.ndarray of shape (n_ts, n_points)
        b_weights: np.ndarray of shape (n_ts, n_points)
        w_*_scaled: float, the weight of the corresponding term
        obstacle (callable): Implicit function representing the obstacle.
        fun_anisotropy_dir (callable): Function that computes the anisotropy direction of the object.
        
    Returns:
        obj: torch.tensor representing a scalar
        grad_g: torch.tensor of shape (n_ts, 7)
        grad_pos_: torch.tensor of shape (n_ts, n_points, 3)
    '''
    obj_fit = compare_last_registration(gs, gt)
    grad_fit_g = grad_compare_last_registration(gs, gt)
    grad_fit_pos_ = np.zeros_like(pos_)
    
    if w_energy_scaled == 0.0:
        obj_energy, grad_energy_g, grad_energy_pos_ = 0.0, 0.0, 0.0
    else:
        obj_energy = energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)
        grad_energy_g, grad_energy_pos_ = grad_energy_path(pos_, gs, masses, a_weights, b_weights, fun_anisotropy_dir)

    obj = w_fit_scaled * obj_fit + w_energy_scaled * obj_energy
    grad_g = w_fit_scaled * grad_fit_g + w_energy_scaled * grad_energy_g
    grad_pos = w_fit_scaled * grad_fit_pos_ + w_energy_scaled * grad_energy_pos_

    return obj, grad_g, grad_pos

class OptimizationBookkeeper:
    def __init__(
        self, n_steps, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, close_gait, 
        vertices_ref, vertex_face_ids, hinge_vertex_ids,
        w_fit_scaled, w_energy_scaled,
        fun_anisotropy_dir, fun_obj_grad_g
    ):
        self.obj_values = []
        self.params_values = []
        self.g_values = []
        self.time_values = [0.0]
        self.start_time = time.time()

        self.n_steps = n_steps
        self.masses = masses
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.force_0 = force_0
        self.torque_0 = torque_0
        self.n_cp = n_cp
        self.close_gait = close_gait
        self.vertices_ref = torch.tensor(vertices_ref)
        self.vertex_face_ids = torch.tensor(vertex_face_ids)
        self.hinge_vertex_ids = torch.tensor(hinge_vertex_ids)
        
        self.w_fit_scaled = w_fit_scaled
        self.w_energy_scaled = w_energy_scaled
        self.gt = gt
        self.fun_obj_grad_g = fun_obj_grad_g
        self.fun_anisotropy_dir = fun_anisotropy_dir
        
    def callback(self, x):
        self.time_values.append(time.time() - self.start_time)
        self.params_values.append(x.tolist())
        params = torch.tensor(x)
        cps_torch = params.reshape(self.n_cp, -1)
        pos_ = sierpinski_deformation_from_cubic_spline(
            cps_torch, self.vertices_ref, self.vertex_face_ids, 
            self.hinge_vertex_ids, self.n_steps, self.n_cp, close_gait=self.close_gait
        )
        pos_np_ = pos_.detach().numpy()
        normals_np_ = self.fun_anisotropy_dir(torch.tensor(pos_np_)).numpy()
        options = {'maxfev': 2000}
        pos, normals, g = multiple_steps_forward(
            pos_np_, normals_np_, self.masses, self.a_weights, self.b_weights, self.force_0, self.torque_0, options=options
        )
        self.g_values.append(g.tolist())
        obj, grad_obj_g, grad_obj_pos_ = self.fun_obj_grad_g(g, pos_np_, self.gt, self.masses, self.a_weights, self.b_weights, self.w_fit_scaled, self.w_energy_scaled, self.fun_anisotropy_dir)
        self.obj_values.append(obj.item())


def main(_):
    
    ###########################################################################
    ## GET QUANTITIES RELATED TO THE TRIAL NUMBER
    ###########################################################################
    
    trial_number = FLAGS.trial_number
    tag_experiment = "_{:02d}".format(trial_number)

    settings_dict = return_sierpinsky_experiment_settings(trial_number)
    n_cp, n_ts = settings_dict["n_cp"], settings_dict["n_ts"]
    rho, eps, close_gait = settings_dict['rho'], settings_dict['eps'], settings_dict['close_gait']
    init_perturb_magnitude = settings_dict['init_perturb_magnitude']
    w_fit, w_energy = settings_dict['w_fit'], settings_dict['w_energy']
    gt = np.array(settings_dict['gt'])
    maxiter = settings_dict['maxiter']

    iterations = 3
    vertices_2D, faces, face_ids, vertex_face_ids, hinge_vertex_ids = sierpinski_triangle_dense(iterations)
    vertices = np.zeros(shape=(vertices_2D.shape[0], 3))
    vertices[:, :2] = vertices_2D
    vertices_ref = vertices
    edges = igl.edges(faces)
    voronoi_area_per_vertex = np.diag(igl.massmatrix(vertices, faces).todense())
    n_points = vertices.shape[0]
    
    aabb = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    aabb_diag = np.linalg.norm(aabb[1] - aabb[0])
    w_fit_scaled = w_fit / aabb_diag ** 2
    w_energy_scaled = w_energy / (rho * aabb_diag * aabb_diag ** 2)

    def fun_anisotropy_dir(pos_):
        '''
        Args:
            pos_: torch.tensor of shape (n_ts, n_points, 3) or (n_points, 3)
            
        Returns:
            normals_: torch.tensor of shape (n_ts, n_points, 3) or (n_points, 3)
        '''
        if pos_.dim() == 2:
            return compute_mesh_vertex_normals(pos_, torch.tensor(faces))
        else:
            return vmap_compute_mesh_vertex_normals(pos_, torch.tensor(faces))
    
    ###########################################################################
    ## GENERATE THE GEOMETRY
    ###########################################################################

    rng = np.random.default_rng(1123)
    params = np.zeros(shape=(n_cp, 3)) # the dihedral angles at each step
    perturb = rng.normal(0.0, 1.0, size=params.shape)
    perturb[0] = 0.0
    perturb = init_perturb_magnitude * perturb / (1.0e-10 + np.linalg.norm(perturb, axis=1, keepdims=True))
    params = (params + perturb).reshape(-1,)
    cps = torch.tensor(params).reshape(n_cp, -1)
    pos_ = sierpinski_deformation_from_cubic_spline(
        cps, torch.tensor(vertices_ref), torch.tensor(vertex_face_ids), 
        torch.tensor(hinge_vertex_ids), n_ts, n_cp, close_gait=close_gait
    ).numpy()

    normals_ = fun_anisotropy_dir(torch.tensor(pos_)).numpy()
    masses = rho * np.tile(voronoi_area_per_vertex.reshape(1, -1), reps=(n_ts, 1))
    a_weights = np.ones(shape=(n_ts, n_points))
    b_weights = (eps - 1.0) * np.ones(shape=(n_ts, n_points))

    force_0 = np.zeros(shape=(3,))
    torque_0 = np.zeros(shape=(3,))
        
    ###########################################################################
    ## MOTION FROM NOTHING
    ###########################################################################

    pos, normals, g = multiple_steps_forward(
        pos_, normals_, masses, a_weights, b_weights, force_0, torque_0, g0=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )

    save_path = os.path.join(path_to_output_sierpinsky, "sierpinsky_init{}.json".format(tag_experiment))

    export_meshes_to_json(
        pos_, g, pos, force_0, torque_0, edges, faces, save_path,
        weights_optim=None, quantities_per_vertex=None,
        quantities_per_edge=None, target_checkpoints_g=gt,
    )
    
    ###########################################################################
    ## OPTIMIZE THE TRAJECTORY
    ###########################################################################

    obj_and_grad_params_scipy = lambda x_: obj_and_grad_params(
        x_, n_ts, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, close_gait,
        vertices_ref, vertex_face_ids, hinge_vertex_ids,
        w_fit_scaled, w_energy_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    
    optim_bookkeeper = OptimizationBookkeeper(
        n_ts, gt, masses, a_weights, b_weights, 
        force_0, torque_0, n_cp, close_gait, 
        vertices_ref, vertex_face_ids, hinge_vertex_ids,
        w_fit_scaled, w_energy_scaled,
        fun_anisotropy_dir, fun_obj_grad_g,
    )
    optim_cb = optim_bookkeeper.callback
    params0 = cps.reshape(-1,).numpy()
    settings_dict['params_init'] = params0.tolist()

    lb = -np.pi * np.ones_like(params0)
    ub = np.pi * np.ones_like(params0)
    bnds = Bounds(lb, ub)

    start_time = time.time()
    result = minimize(
        obj_and_grad_params_scipy, params0, bounds=bnds, jac=True, 
        method='L-BFGS-B', options={'disp': True, 'maxiter': maxiter, 'ftol': 1.0e-10, 'gtol': 1.0e-6},
        callback=optim_cb,
    )
    optim_duration = time.time() - start_time
    
    ###########################################################################
    ## RUN THE OPTIMAL FORWARD PASS AND SAVE THE RESULTS
    ###########################################################################
    
    params_opt = torch.tensor(result.x.copy())
    settings_dict['params_opt'] = params_opt.tolist()
    pos_opt_ = sierpinski_deformation_from_cubic_spline(
        params_opt.reshape(-1, 3), torch.tensor(vertices_ref), torch.tensor(vertex_face_ids), 
        torch.tensor(hinge_vertex_ids), n_ts, n_cp, close_gait=close_gait
    )
    pos_opt_ = pos_opt_.detach().numpy()
    normals_opt_ = fun_anisotropy_dir(torch.tensor(pos_opt_)).numpy()

    options = {'maxfev': 2000}
    pos_opt, normals_opt, g_opt = multiple_steps_forward(
        pos_opt_, normals_opt_, masses, a_weights, b_weights, force_0, torque_0, options=options
    )

    save_path_opt = os.path.join(path_to_output_sierpinsky, "sierpinsky_opt{}.json".format(tag_experiment))

    weights_optim = {
        'w_fit': w_fit,
        'w_fit_scaled': w_fit_scaled,
        'w_energy': w_energy,
        'w_energy_scaled': w_energy_scaled,
    }

    optim_evol_data = {
        'obj_values': optim_bookkeeper.obj_values,
        'optim_duration': optim_duration,
        'params_values': optim_bookkeeper.params_values,
        'g_values': optim_bookkeeper.g_values,
        'time_values': optim_bookkeeper.time_values,
    }

    export_meshes_to_json(
        pos_opt_, g_opt, pos_opt, force_0, torque_0, edges, faces, save_path_opt,
        weights_optim=weights_optim, quantities_per_vertex=None,
        quantities_per_edge=None, target_checkpoints_g=gt,
        optimization_settings=settings_dict, optimization_duration=optim_duration,
        optimization_evolution=optim_evol_data,
    )

    
    print("Optimization results:")
    print("Final position:", g_opt[-1, 4:])
    print_quaternion(torch.tensor(g[-1, :4]))
    print_quaternion(torch.tensor(g_opt[-1, :4]))
    print_quaternion(torch.tensor(gt[:4]))


if __name__ == '__main__':
    app.run(main)