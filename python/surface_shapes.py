import os
import sys as _sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CUBICSPLINES = os.path.join(os.path.dirname(SCRIPT_PATH), 'ext/torchcubicspline')
_sys.path.append(PATH_TO_CUBICSPLINES)

import torch
from torchcubicspline import (
    natural_cubic_spline_coeffs, NaturalCubicSpline
)
from utils import register_points_torch, vmap_align_point_cloud_no_flips, align_point_clouds

def factorize_laplacian(laplacian, mass_matrix, free_indices, fixed_indices, harmonic_order=2):
    '''
    Args:
        laplacian: torch.tensor of shape (N, N)
        mass_matrix: torch.tensor of shape (N, N), the dense mass matrix
        free_indices: list of length N - K containing all the free indices corresponding to the unknown values
        fixed_indices: list of length K containing all the fixed indices corresponding to the known values
        harmonic_order: int
        
        
    Returns:
        Luu: torch.tensor of shape (N - K, N - K), the Cholesky factor of the free part of the Laplacian
        Quk: torch.tensor of shape (N - K, K), the block of the Laplacian corresponding to the free indices and the fixed indices
    '''
    
    Q = - laplacian
    mass_matrix_inv = torch.inverse(mass_matrix)
    scaled_laplacian = - torch.mm(mass_matrix_inv, laplacian)

    for i in range(1, harmonic_order):
        Q = torch.mm(Q, scaled_laplacian)
    
    Quu = Q[free_indices][:, free_indices]
    Luu = torch.linalg.cholesky(Quu) # does not work with sparse matrices
    Quk = Q[free_indices][:, fixed_indices]
    
    return Luu, Quk

def harmonic_torch(Luu, Quk, free_indices, fixed_indices, fixed_values):
    '''
    Solves the linear system Luu * free_values = - Quk * fixed_values, and returns the concatenated solution
    
    Args:
        Luu: torch.tensor of shape (N - K, N - K), the Cholesky factor of the free part of the Laplacian
        Quk: torch.tensor of shape (N - K, K), the block of the Laplacian corresponding to the free indices and the fixed indices
        free_indices: list of length N - K containing all the free indices corresponding to the unknown values
        fixed_indices: list of length K containing all the fixed indices corresponding to the known values
        fixed_values: torch.tensor of shape (K, 3) containing the known values
    
    Returns:
        sol: torch.tensor of shape (N, 3)
        
    Note:
        The implementation allows for batch processing, i.e. the input tensors can have an arbitrary number of dimensions
    ''' 

    rhs = - Quk @ fixed_values
    batch_size = list(fixed_values.shape)[:-2]
    values_size = batch_size + [len(free_indices)+len(fixed_indices), fixed_values.shape[-1]]
    values = torch.zeros(size=values_size, dtype=torch.float64)
    values[..., free_indices, :] = torch.cholesky_solve(rhs, Luu) # does not work with sparse matrices
    values[..., fixed_indices, :] = fixed_values
    return values

def transform_vertices_from_handle_deformations(handle_defos, Luu, Quk, free_ids, handles_ids, vertices_ref, vertices_ref_rot):
    '''
    Transforms the vertices from the handle deformations
    
    Args:
        handle_defos: torch.tensor of shape (?, n_handles, 3), the handle deformations
        vertices_ref: torch.tensor of shape (n_vertices, 3), the reference vertices
        vertices_ref_rot: torch.tensor of shape (n_vertices, 3), the reference vertices after for registration (preferably perturbed versions of vertices_ref, should be different from vertices_ref to avoid singularities)
        Same as above for Luu, Quk, free_ids, handles_ids
        
    Returns:
        new_vertices: torch.tensor of shape (?, n_vertices, 3), the transformed vertices after centering
    '''
    d_vertices = harmonic_torch(Luu, Quk, free_ids, handles_ids, handle_defos)
    vertices_ref_rot_rep = vertices_ref_rot.unsqueeze(0).repeat(d_vertices.shape[0], 1, 1)
    new_vertices = register_points_torch(vertices_ref.unsqueeze(0) + d_vertices, vertices_ref_rot_rep, allow_flip=False)
    return new_vertices

def handle_deformations_generation_cubic_splines(
    control_points, Luu, Quk, free_ids, handles_ids, vertices_ref, vertices_ref_rot, handle_defos_init, free_handle_dofs, n_ts, close_motion=False,
):
    '''Generate a deformation in a linear subspace and deforms the original mesh accordingly
    
    Args:
        control_points: (n_cp, n_free_dofs) tensor representing the control points
        Same as above for Luu, Quk, free_ids, handles_ids, vertices_ref
        handles_init: (n_handles, 3) tensor representing the initial positions of the handles
        free_handle_dofs: list of ints representing the indices of the free handles
        n_ts: int representing the number of time steps
        close_motion: bool representing whether the motion is closed or not

    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    n_cp, n_free_dofs = control_points.shape
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_motion)

    if close_motion:
        control_points = torch.cat([control_points, control_points[0].reshape(1, n_free_dofs)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points.reshape(n_cp+close_motion, -1))
    spline = NaturalCubicSpline(spline_coeffs)
    free_handle_defos = spline.evaluate(t).reshape(n_ts, -1)
    all_handle_defos = handle_defos_init.clone().reshape(n_ts, -1)
    all_handle_defos[:, free_handle_dofs] = free_handle_defos
    all_handle_defos = all_handle_defos.reshape(n_ts, -1, 3)
    
    pos_ = transform_vertices_from_handle_deformations(all_handle_defos, Luu, Quk, free_ids, handles_ids, vertices_ref, vertices_ref_rot).reshape(n_ts, -1, 3)

    return pos_

def deformations_generation_cubic_splines(
    control_points, vertices_ref, vertices_ref_rot, defos_init, free_vertex_dofs, n_ts, close_motion=False,
):
    '''Generate a deformation in a linear subspace and deforms the original mesh accordingly
    
    Args:
        control_points: (n_cp, n_free_dofs) tensor representing the control points
        Same as above for Luu, Quk, free_ids, handles_ids, vertices_ref
        handles_init: (n_handles, 3) tensor representing the initial positions of the handles
        free_vertex_dofs: list of ints representing the indices of the free handles
        n_ts: int representing the number of time steps
        close_motion: bool representing whether the motion is closed or not

    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    n_cp, n_free_dofs = control_points.shape
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_motion)

    if close_motion:
        control_points = torch.cat([control_points, control_points[0].reshape(1, n_free_dofs)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points.reshape(n_cp+close_motion, -1))
    spline = NaturalCubicSpline(spline_coeffs)
    free_defos = spline.evaluate(t).reshape(n_ts, -1)
    all_defos = defos_init.clone().reshape(n_ts, -1)
    all_defos[:, free_vertex_dofs] = free_defos
    all_defos = all_defos.reshape(n_ts, -1, 3)
    
    vertices_ref_rot_rep = vertices_ref_rot.unsqueeze(0).repeat(all_defos.shape[0], 1, 1)
    pos_ = register_points_torch(vertices_ref.unsqueeze(0) + all_defos, vertices_ref_rot_rep, allow_flip=False)

    return pos_

def project_into_subspace(vertices, modes, mass, vertices_ref):
    '''Projects the vertices onto the linear subspace spanned by the modes
    Args:
        vertices: torch.tensor of shape (n_vertices, 3), the vertices to project onto the affine subspace
        modes: torch.tensor of shape (n_vertices, n_modes)
        mass: torch.tensor of shape (n_vertices, n_vertices), the mass matrix
        vertices_ref: torch.tensor of shape (n_vertices, 3), the reference vertices
        
    Returns:
        projected_vertices: torch.tensor of shape (n_vertices, 3), the projected vertices
    '''
    deformations = vertices - vertices_ref
    return modes.T @ (mass @ deformations)

vmap_project_into_subspace = torch.vmap(project_into_subspace, in_dims=(0, None, None, None))

def transform_vertices_from_subspace(modal_coeffs, modes, vertices_ref):
    '''Transforms the vertices from the linear subspace spanned by the modes
    Args:
        vertices_ref: torch.tensor of shape (n_vertices, 3), the reference vertices
        modes: torch.tensor of shape (n_vertices, n_modes)
        modal_coeffs: torch.tensor of shape (n_modes, 3)
        
    Returns:
        vertices: torch.tensor of shape (n_vertices, 3), the transformed vertices
    '''
    return vertices_ref + modes @ modal_coeffs

vmap_transform_vertices_from_subspace = torch.vmap(transform_vertices_from_subspace, in_dims=(0, None, None))

def laplacian_modes_generation_cubic_splines(
    control_points, modes, vertices_ref, n_ts, close_motion=False, vertices_ref_rot=None,
):
    '''Generate a deformation in a linear subspace and deforms the original mesh accordingly
    
    Args:
        control_points: (n_cp, n_modes, 3) tensor representing the control points
        modes: (n_vertices, n_modes) tensor representing the laplacian eigenmodes
        vertices_ref: (n_vertices, 3) tensor representing the reference vertices
        n_ts: int representing the number of time steps
        close_motion: bool representing whether the motion is closed or not

    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    n_cp, n_modes, _ = control_points.shape
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_motion)

    if close_motion:
        control_points = torch.cat([control_points, control_points[0].reshape(1, n_modes, 3)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points.reshape(n_cp+close_motion, -1))
    spline = NaturalCubicSpline(spline_coeffs)
    modal_coeffs = spline.evaluate(t).reshape(n_ts, n_modes, 3)

    if vertices_ref_rot is not None:
        pos_ = register_points_torch(
            vmap_transform_vertices_from_subspace(modal_coeffs, modes, vertices_ref).reshape(n_ts, -1, 3), vertices_ref_rot.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False
        )
    else:
        pos_ = vmap_transform_vertices_from_subspace(modal_coeffs, modes, vertices_ref).reshape(n_ts, -1, 3)

    return pos_

def project_into_subspace_anisotropic(vertices, modes, mass, vertices_ref):
    '''Projects the vertices onto the linear subspace spanned by the modes
    Args:
        vertices: torch.tensor of shape (n_vertices, 3), the vertices to project onto the affine subspace
        modes: torch.tensor of shape (3*n_vertices, n_modes)
        mass: torch.tensor of shape (3*n_vertices, 3*n_vertices), the mass matrix
        vertices_ref: torch.tensor of shape (n_vertices, 3), the reference vertices
        
    Returns:
        projected_vertices: torch.tensor of shape (n_vertices, 3), the projected vertices
    '''
    deformations = vertices - vertices_ref
    return modes.T @ (mass @ deformations.reshape(-1,)).reshape(-1, 3)

vmap_project_into_subspace_anisotropic = torch.vmap(project_into_subspace_anisotropic, in_dims=(0, None, None, None))

def transform_vertices_from_subspace_anisotropic(modal_coeffs, modes, vertices_ref):
    '''Transforms the vertices from the linear subspace spanned by the modes
    Args:
        vertices_ref: torch.tensor of shape (n_vertices, 3), the reference vertices
        modes: torch.tensor of shape (3*n_vertices, n_modes)
        modal_coeffs: torch.tensor of shape (n_modes, 3)
        
    Returns:
        vertices: torch.tensor of shape (n_vertices, 3), the transformed vertices
    '''
    return vertices_ref + (modes @ modal_coeffs).reshape(-1, 3)

vmap_transform_vertices_from_subspace_anisotropic = torch.vmap(transform_vertices_from_subspace_anisotropic, in_dims=(0, None, None))

def elasticity_hessian_modes_generation_cubic_splines(
    control_points, modes, vertices_ref, n_ts, close_motion=False,
):
    '''Generate a deformation in a linear subspace and deforms the original mesh accordingly
    
    Args:
        control_points: (n_cp, n_modes) tensor representing the control points
        modes: (n_vertices, n_modes) tensor representing the laplacian eigenmodes
        vertices_ref: (n_vertices, 3) tensor representing the reference vertices
        n_ts: int representing the number of time steps
        close_motion: bool representing whether the motion is closed or not

    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    n_cp, n_modes = control_points.shape
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_motion)

    if close_motion:
        control_points = torch.cat([control_points, control_points[0].reshape(1, n_modes)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points.reshape(n_cp+close_motion, -1))
    spline = NaturalCubicSpline(spline_coeffs)
    modal_coeffs = spline.evaluate(t).reshape(n_ts, n_modes)
    
    pos_ = vmap_transform_vertices_from_subspace_anisotropic(modal_coeffs, modes, vertices_ref).reshape(n_ts, -1, 3)

    return pos_
