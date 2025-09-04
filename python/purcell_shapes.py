import os
import sys as _sys
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CUBICSPLINES = os.path.join(os.path.dirname(SCRIPT_PATH), 'ext/torchcubicspline')
_sys.path.append(PATH_TO_CUBICSPLINES)

import torch
from torchcubicspline import (
    natural_cubic_spline_coeffs, NaturalCubicSpline
)
from utils import register_points_torch, vmap_rotate_about_axis


def interpolate_hinges(hinge_vertices, n_pts_per_segment, edges):
    '''
    Generate an interpolated sequence of hinge positions of a swimmer, each parameterized by two scalar angles
    alpha = torch.tensor([alpha_1, alpha_2]) in [-Pi, Pi] with fixed segment ratios.

    Args:
        hinge_vertices: (n_ts, n_hinges, 3) tensor representing the hinge positions of the Purcell swimmer.
        n_pts_per_segment: Int, number of points to interpolate per segment.
        edges: (n_edges, 2) tensor of the edges to interpolate.

    Returns:
        torch.Tensor of shape (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) representing the interpolated vertex positions in 2D (x, y, 0 for z-axis)
    '''

    used_hinges = hinge_vertices[:, torch.unique(edges)]
    n_used_edges = used_hinges.shape[1]
    n_edges = edges.shape[0]
    ts = torch.linspace(0.0, 1.0, n_pts_per_segment + 2)[1:-1].reshape(1, 1, -1, 1)
    interpolated_vertices = torch.zeros(size=(hinge_vertices.shape[0], n_edges*(n_pts_per_segment) + n_used_edges, 3))
    interpolated_vertices[:, :-n_used_edges, :] = (hinge_vertices[:, edges[:, 0]].unsqueeze(2) + ts * (hinge_vertices[:, edges[:, 1]] - hinge_vertices[:, edges[:, 0]]).unsqueeze(2)).reshape(hinge_vertices.shape[0], -1, 3)
    interpolated_vertices[:, -n_used_edges:] = used_hinges
    
    return interpolated_vertices

def generate_control_points(params, close_gait=True):
    '''
    Generic function to generate control points for a swimmer given parameters.

    Args:
        params: (n_cps, n_dim) tensor of the control points.
        close_gait: Bool, whether to close the gait by appending the first point at the end.
    
    Returns:
        Tensor: (n_cps+close_grait, n_dim) tensor representing the control points.
    '''
    n_cp, n_dim = params.shape
    
    # Close the gait by appending the first point at the end
    if close_gait:
        all_cps = torch.zeros(n_cp + 1, n_dim)
        all_cps[:-1] = params
        all_cps[-1] = params[0]
    else:
        all_cps = params
    return all_cps

def purcell_shape_hinges(alpha, e1bye2=1.0, e3bye2=1.0, total_length=1.0):
    '''
    Generate an seququence of hinge positions of a Purcell swimmer, each parameterized by two scalar angles
    alpha = torch.tensor([alpha_1, alpha_2]) in [-Pi, Pi] with fixed segment ratios.

    Args:
        alpha: (n_ts, 2) tensor of two angles alpha = [alpha_1, alpha_2] in radians.
        e1bye2: Float, ratio of the length of segment 1 to segment 2.
        e3bye2: Float, ratio of the length of segment 3 to segment 2.
        total_length: Float, total length of the Purcell swimmer.

    Returns:
        torch.Tensor of shape (n_ts, 4, 3) representing the interpolated vertex positions in 2D (x, y, 0 for z-axis) normalized to have total length 1.

    The Purcell swimmer is a 2D swimmer with 4 vertices, with the following shape:
           0          3
            o        o
             \      /
     alpha_1  o----o  alpha_2
              1    2
    '''

    # Initialize vertices tensor
    hinge_vertices = torch.zeros(size=(alpha.shape[0], 4, 3))

    # Fixed segment ratios
    length_e2 = total_length / (1.0 + e1bye2 + e3bye2)

    # Define base vertices based on alpha angles
    hinge_vertices[:, 1, 0] = - length_e2 / 2.0
    hinge_vertices[:, 2, 0] = length_e2 / 2.0
    hinge_vertices[:, 0, 0] = hinge_vertices[:, 1, 0] - e1bye2 * length_e2 * torch.cos(-alpha[:, 0])
    hinge_vertices[:, 0, 1] = hinge_vertices[:, 1, 1] - e1bye2 * length_e2 * torch.sin(-alpha[:, 0])
    hinge_vertices[:, 3, 0] = hinge_vertices[:, 2, 0] + e3bye2 * length_e2 * torch.cos(alpha[:, 1])
    hinge_vertices[:, 3, 1] = hinge_vertices[:, 2, 1] + e3bye2 * length_e2 * torch.sin(alpha[:, 1])
    
    return hinge_vertices

def purcell_generate_fun_anisotropy_dir(n_pts_per_segment):
    '''
    Generate a function that returns the anisotropy direction of the Purcell swimmer at each point.

    Args:
        n_pts_per_segment: Int, number of points to interpolate per segment.
        edges: (n_edges, 2) tensor of the edges to interpolate.
    
    Returns:
        Function: A function that returns the anisotropy direction of the Purcell swimmer at each point.
    '''

    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    n_edges = edges.shape[0]
    n_used_hinges = torch.unique(edges).shape[0]
    offset_edges = edges + n_edges * n_pts_per_segment
    id_pos_to_edge = torch.zeros(n_edges * n_pts_per_segment + n_used_hinges, dtype=torch.long)
    id_pos_to_edge[:n_edges * n_pts_per_segment] = torch.repeat_interleave(torch.arange(n_edges), n_pts_per_segment, dim=0)
    id_pos_to_edge[n_edges * n_pts_per_segment    ] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 1] = 1
    id_pos_to_edge[n_edges * n_pts_per_segment + 2] = 1
    id_pos_to_edge[n_edges * n_pts_per_segment + 3] = 2
    
    def fun_anisotropy_dir(pos):
        '''
        Args:
            hinge_vertices: (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) tensor representing the positions of the symmetric Purcell swimmer which has been generated using the symmetric_purcell_shape_hinges function.

        Returns:
            torch.Tensor of shape (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) representing the interpolated anisotropy directions.
        '''
        dir_edge = pos[..., offset_edges[:, 1], :] - pos[..., offset_edges[:, 0], :]
        dir_edge = dir_edge / torch.norm(dir_edge, dim=-1, keepdim=True) # shape: (n_ts, n_edges, 3)
        interpolated_anisotropy_dirs = dir_edge[..., id_pos_to_edge, :] # shape: (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3)
        
        return interpolated_anisotropy_dirs

    return fun_anisotropy_dir

def purcell_generate_control_points_polar(params, close_gait=True):
    '''
    Generate control points for a Purcell swimmer given parameters in the [alpha_1, alpha_2] space using polar coordinates.

    Args:
        params: (n_cps, 2) tensor of radii and phis of the control points in the [alpha_1, alpha_2] space.
        close_gait: Bool, whether to close the gait by appending the first point at the end.
    
    Returns:
        Tensor: (n_cps+close_gait, 2) tensor representing the radii and phis of the control points in the [alpha_1, alpha_2] space.
    '''
    n_cp = params.shape[0]
    radii = params[:, 0]
    phis = params[:, 1]
    
    # Initialize alphas with fixed polar angles that vary over time
    control_points = torch.zeros(n_cp, 2)
    control_points[:, 0] = radii * torch.cos(2 * torch.pi * phis)
    control_points[:, 1] = radii * torch.sin(2 * torch.pi * phis)
    
    # Close the gait by appending the first point at the end
    if close_gait:
        all_cps = torch.zeros(n_cp + 1, 2)
        all_cps[:-1] = control_points
        all_cps[-1] = control_points[0]
    else:
        all_cps = control_points
    return all_cps

def purcell_generator_polar_cubic_splines(params, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with varying polar angles.

    Args:
        params: (n_cps, 2) tensor of radii and phis of the control points in the [alpha_1, alpha_2] space.
        n_ts: Int, number of time steps to interpolate the Purcell swimmer.
        n_pts_per_segment: Int, number of points to interpolate per segment.
        e1bye2: Float, ratio of the length of segment 1 to segment 2.
        e3bye2: Float, ratio of the length of segment 3 to segment 2.
        total_length: Float, total length of the Purcell swimmer.
        close_gait: Bool, whether to close the gait by appending the first point at the end.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    n_cp = params.shape[0]
    t = torch.linspace(0.0, 1.0, n_ts)
    # t = torch.linspace(0.0, 1.0, n_ts + 1)[:-1]
    ts_cp = torch.linspace(0.0, 1.0, n_cp + close_gait)
    
    all_cps = purcell_generate_control_points_polar(params, close_gait=True)

    # interpolate spline
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, all_cps, close_spline=close_gait)
    spline = NaturalCubicSpline(spline_coeffs)
    ws = spline.evaluate(t)
    
    hinge_vertices = purcell_shape_hinges(ws, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges)

def purcell_generator_polar_fixed_angles_cubic_splines(radii, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with fixed polar angles.

    Args:
        radii: (n_cps,) tensor of radii of the control points.
        Same as purcell_generator_polar_cubic_splines, but with fixed polar angles.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    params = torch.zeros(len(radii), 2)
    params[:, 0] = radii
    params[:, 1] = torch.linspace(0.0, 1.0, len(radii)+1)[:-1]
    return purcell_generator_polar_cubic_splines(params, n_ts, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait)

def purcell_generator_polar_fixed_angles_cubic_splines_origin_symmetry(radii, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with fixed polar angles.

    Args:
        radii: (n_cps,) tensor of radii of the control points, to be symmetrized about the origin.
        Same as purcell_generator_polar_cubic_splines, but with fixed polar angles.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    radii_symmetric = torch.cat([radii, radii], dim=0)
    return purcell_generator_polar_fixed_angles_cubic_splines(radii_symmetric, n_ts, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait)

def symmetric_purcell_shape_hinges(alpha, e1bye2=1.0, e3bye2=1.0, total_length=1.0):
    '''
    Generate an seququence of hinge positions of a Purcell swimmer, each parameterized by two scalar angles
    alpha = torch.tensor([alpha_1, alpha_2]) in [-Pi, Pi] with fixed segment ratios.

    Args:
        alpha: (n_ts, 2) tensor of two angles alpha = [alpha_1, alpha_2] in radians.
        e1bye2: Float, ratio of the length of segment 1 to segment 2.
        e3bye2: Float, ratio of the length of segment 3 to segment 2.
        total_length: Float, total length of the Purcell swimmer.

    Returns:
        torch.Tensor of shape (n_ts, 6, 3) representing the interpolated vertex positions in 2D (x, y, 0 for z-axis) normalized to have total length 1.

    The symmetric Purcell swimmer is a 2D swimmer with 6 vertices, with the following shape:
           2          3
            o        o
     alpha_1 \      / alpha_2
              o----o 
             /0    1\ 
            o        o
           4          5
    '''

    # Initialize vertices tensor
    hinge_vertices = torch.zeros(size=(alpha.shape[0], 6, 3))

    # Fixed segment ratios
    length_e2 = total_length / (1.0 + e1bye2 + e3bye2)

    # Define base vertices based on alpha angles
    hinge_vertices[:, 0, 0] = - length_e2 / 2.0
    hinge_vertices[:, 1, 0] = length_e2 / 2.0
    hinge_vertices[:, 2, 0] = hinge_vertices[:, 0, 0] - e1bye2 * length_e2 * torch.cos(-alpha[:, 0])
    hinge_vertices[:, 2, 1] = hinge_vertices[:, 0, 1] - e1bye2 * length_e2 * torch.sin(-alpha[:, 0])
    hinge_vertices[:, 3, 0] = hinge_vertices[:, 1, 0] + e3bye2 * length_e2 * torch.cos(alpha[:, 1])
    hinge_vertices[:, 3, 1] = hinge_vertices[:, 1, 1] + e3bye2 * length_e2 * torch.sin(alpha[:, 1])
    hinge_vertices[:, 4, 0] = hinge_vertices[:, 0, 0] - e1bye2 * length_e2 * torch.cos(alpha[:, 0])
    hinge_vertices[:, 4, 1] = hinge_vertices[:, 0, 1] - e1bye2 * length_e2 * torch.sin(alpha[:, 0])
    hinge_vertices[:, 5, 0] = hinge_vertices[:, 1, 0] + e3bye2 * length_e2 * torch.cos(-alpha[:, 1])
    hinge_vertices[:, 5, 1] = hinge_vertices[:, 1, 1] + e3bye2 * length_e2 * torch.sin(-alpha[:, 1])
    
    return hinge_vertices

def symmetric_purcell_generate_fun_anisotropy_dir(n_pts_per_segment, edges):
    '''
    Generate a function that returns the anisotropy direction of the Purcell swimmer at each point.

    Args:
        n_pts_per_segment: Int, number of points to interpolate per segment.
        edges: (n_edges, 2) tensor of the edges to interpolate.
    
    Returns:
        Function: A function that returns the anisotropy direction of the Purcell swimmer at each point.
    '''

    n_edges = edges.shape[0]
    n_used_hinges = torch.unique(edges).shape[0]
    offset_edges = edges + n_edges * n_pts_per_segment
    id_pos_to_edge = torch.zeros(n_edges * n_pts_per_segment + n_used_hinges, dtype=torch.long)
    id_pos_to_edge[:n_edges * n_pts_per_segment] = torch.repeat_interleave(torch.arange(n_edges), n_pts_per_segment, dim=0)
    id_pos_to_edge[n_edges * n_pts_per_segment    ] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 1] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 2] = 1
    id_pos_to_edge[n_edges * n_pts_per_segment + 3] = 2
    id_pos_to_edge[n_edges * n_pts_per_segment + 4] = 3
    id_pos_to_edge[n_edges * n_pts_per_segment + 5] = 4
    
    def fun_anisotropy_dir(pos):
        '''
        Args:
            hinge_vertices: (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) tensor representing the positions of the symmetric Purcell swimmer which has been generated using the symmetric_purcell_shape_hinges function.

        Returns:
            torch.Tensor of shape (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) representing the interpolated anisotropy directions.
        '''
        dir_edge = pos[..., offset_edges[:, 1], :] - pos[..., offset_edges[:, 0], :]
        dir_edge = dir_edge / torch.norm(dir_edge, dim=-1, keepdim=True) # shape: (n_ts, n_edges, 3)
        interpolated_anisotropy_dirs = dir_edge[..., id_pos_to_edge, :] # shape: (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3)
        
        return interpolated_anisotropy_dirs

    return fun_anisotropy_dir
    

def symmetric_purcell_generator_polar_cubic_splines(params, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with varying polar angles.

    Args:
        params: (n_cps, 2) tensor of radii and phis of the control points in the [alpha_1, alpha_2] space.
        n_ts: Int, number of time steps to interpolate the Purcell swimmer.
        n_pts_per_segment: Int, number of points to interpolate per segment.
        e1bye2: Float, ratio of the length of segment 1 to segment 2.
        e3bye2: Float, ratio of the length of segment 3 to segment 2.
        total_length: Float, total length of the Purcell swimmer.
        close_gait: Bool, whether to close the gait by appending the first point at the end.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    n_cp = params.shape[0]
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp + close_gait)
    
    all_cps = purcell_generate_control_points_polar(params, close_gait=True)

    # interpolate spline
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, all_cps, close_spline=close_gait)
    spline = NaturalCubicSpline(spline_coeffs)
    ws = spline.evaluate(t)
    
    hinge_vertices = symmetric_purcell_shape_hinges(ws, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length)
    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5]])
    return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges)

def symmetric_purcell_generator_polar_fixed_angles_cubic_splines(radii, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with fixed polar angles.

    Args:
        radii: (n_cps,) tensor of radii of the control points.
        Same as purcell_generator_polar_cubic_splines, but with fixed polar angles.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    params = torch.zeros(len(radii), 2)
    params[:, 0] = radii
    params[:, 1] = torch.linspace(0.0, 1.0, len(radii)+1)[:-1]
    return symmetric_purcell_generator_polar_cubic_splines(params, n_ts, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait)

def symmetric_purcell_generator_polar_fixed_angles_cubic_splines_origin_symmetry(radii, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with fixed polar angles.

    Args:
        radii: (n_cps,) tensor of radii of the control points, to be symmetrized about the origin.
        Same as purcell_generator_polar_cubic_splines, but with fixed polar angles.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    radii_symmetric = torch.cat([radii, radii], dim=0)
    return symmetric_purcell_generator_polar_fixed_angles_cubic_splines(radii_symmetric, n_ts, n_pts_per_segment, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length, close_gait=close_gait)


def symmetric_purcell_generator_euclidean_cubic_splines(params, n_ts, n_pts_per_segment, e1bye2=1.0, e3bye2=1.0, total_length=1.0, close_gait=True):
    '''
    Generate a Purcell swimmer that evolves over time with varying polar angles.

    Args:
        params: (n_cps, 2) tensor of radii and phis of the control points in the [alpha_1, alpha_2] space.
        n_ts: Int, number of time steps to interpolate the Purcell swimmer.
        n_pts_per_segment: Int, number of points to interpolate per segment.
        e1bye2: Float, ratio of the length of segment 1 to segment 2.
        e3bye2: Float, ratio of the length of segment 3 to segment 2.
        total_length: Float, total length of the Purcell swimmer.
        close_gait: Bool, whether to close the gait by appending the first point at the end.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
    '''
    n_cp = params.shape[0]
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp + close_gait)
    
    all_cps = generate_control_points(params, close_gait=True)

    # interpolate spline
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, all_cps, close_spline=close_gait)
    spline = NaturalCubicSpline(spline_coeffs)
    ws = spline.evaluate(t)
    
    hinge_vertices = symmetric_purcell_shape_hinges(ws, e1bye2=e1bye2, e3bye2=e3bye2, total_length=total_length)
    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5]])
    return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges)

def turtle_shape_hinges(alpha, beta, lengths):
    '''
    Generate an sequence of hinge positions of a turtle, each parameterized by four scalar angles

    Args:
        alpha: (n_ts, 4) tensor of four angles
        beta: (n_ts, 4) tensor of four angles
        lengths: (n_ts, 9) tensor of the lengths of the segments (see below for the ordering)

    Returns:
        torch.Tensor of shape (n_ts, 10, 3) representing the interpolated vertex positions in 2D (x, y, 0 for z-axis) normalized to have total length 1.

    The turtle shape is a 2D swimmer with 10 vertices, with the following shape:

      6    2          3    7
       o----o        o----o
     alpha_1 \      / alpha_2
              o----o 
     alpha_3 /0    1\ alpha_4
       o----o        o----o
      8    4          5    9
      
    The edge lengths are ordered as [e01, e02, e13, e04, e15, e26, e37, e48, e59].
    The betas are the **turning angles** defined at the following vertices:

      6   b1 2      3 b2    7
       o----o        o----o
             \      / 
              o----o 
             /0    1\ 
       o----o        o----o
      8   b3 4      5 b4    9
    
    '''

    # Initialize vertices tensor
    hinge_vertices = torch.zeros(size=(alpha.shape[0], 10, 3))

    # Define base vertices based on alpha angles
    hinge_vertices[:, 0, 0] = - lengths[0] / 2.0
    hinge_vertices[:, 1, 0] = lengths[0] / 2.0

    hinge_vertices[:, 2, 0] = hinge_vertices[:, 0, 0] + lengths[1] * torch.cos(torch.pi - alpha[:, 0])
    hinge_vertices[:, 2, 1] = hinge_vertices[:, 0, 1] + lengths[1] * torch.sin(torch.pi - alpha[:, 0])
    hinge_vertices[:, 3, 0] = hinge_vertices[:, 1, 0] + lengths[2] * torch.cos(alpha[:, 1])
    hinge_vertices[:, 3, 1] = hinge_vertices[:, 1, 1] + lengths[2] * torch.sin(alpha[:, 1])
    hinge_vertices[:, 4, 0] = hinge_vertices[:, 0, 0] + lengths[3] * torch.cos(torch.pi - alpha[:, 2])
    hinge_vertices[:, 4, 1] = hinge_vertices[:, 0, 1] + lengths[3] * torch.sin(torch.pi - alpha[:, 2])
    hinge_vertices[:, 5, 0] = hinge_vertices[:, 1, 0] + lengths[4] * torch.cos(alpha[:, 3])
    hinge_vertices[:, 5, 1] = hinge_vertices[:, 1, 1] + lengths[4] * torch.sin(alpha[:, 3])
    
    hinge_vertices[:, 6, 0] = hinge_vertices[:, 2, 0] + lengths[5] * torch.cos(torch.pi - alpha[:, 0] + beta[:, 0])
    hinge_vertices[:, 6, 1] = hinge_vertices[:, 2, 1] + lengths[5] * torch.sin(torch.pi - alpha[:, 0] + beta[:, 0])
    hinge_vertices[:, 7, 0] = hinge_vertices[:, 3, 0] + lengths[6] * torch.cos(alpha[:, 1] + beta[:, 1])
    hinge_vertices[:, 7, 1] = hinge_vertices[:, 3, 1] + lengths[6] * torch.sin(alpha[:, 1] + beta[:, 1])
    hinge_vertices[:, 8, 0] = hinge_vertices[:, 4, 0] + lengths[7] * torch.cos(torch.pi - alpha[:, 2] + beta[:, 2])
    hinge_vertices[:, 8, 1] = hinge_vertices[:, 4, 1] + lengths[7] * torch.sin(torch.pi - alpha[:, 2] + beta[:, 2])
    hinge_vertices[:, 9, 0] = hinge_vertices[:, 5, 0] + lengths[8] * torch.cos(alpha[:, 3] + beta[:, 3])
    hinge_vertices[:, 9, 1] = hinge_vertices[:, 5, 1] + lengths[8] * torch.sin(alpha[:, 3] + beta[:, 3])
    
    return hinge_vertices

def turtle_fixed_arms_generator_cubic_splines(params, n_ts, n_pts_per_segment, beta, lengths, close_gait=True, return_discretized=False):
    '''
    Generate a turtle that evolves over time with varying angles.

    Args:
        params: (n_cps, 4) tensor of alpha angles (see turtle_shape_hinges).
        n_ts: Int, number of time steps to interpolate the Purcell swimmer.
        n_pts_per_segment: Int, number of points to interpolate per segment.
        beta: (n_ts, 4) tensor of four angles (see turtle_shape_hinges).
        lengths: (n_ts, 9) tensor of the lengths of the segments (see turtle_shape_hinges for the ordering).
        close_gait: Bool, whether to close the gait by appending the first point at the end.
        return_discretized: Bool, whether to return the discretized spline or not.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
        ws: (n_ts, 4) tensor representing the discretized angles of the swimmer over time.
    '''
    n_cp = params.shape[0]
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp + close_gait)
    
    all_cps = generate_control_points(params, close_gait=True)

    # interpolate spline
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, all_cps, close_spline=close_gait)
    spline = NaturalCubicSpline(spline_coeffs)
    ws = spline.evaluate(t)
    
    hinge_vertices = turtle_shape_hinges(ws, beta.reshape(1, -1), lengths)
    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])
    if return_discretized:
        return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges), ws
    else:
        return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges)
    
def turtle_varying_masses_generator_cubic_splines(params, n_ts, n_pts_per_segment, beta, lengths, close_gait=True, return_discretized=False):
    '''
    Generate a turtle that evolves over time with varying angles.

    Args:
        params: (n_cps, 4 + 9) tensor of alpha angles (see turtle_shape_hinges) and logits representing the mass associated to each segment.
        n_ts: Int, number of time steps to interpolate the Purcell swimmer.
        n_pts_per_segment: Int, number of points to interpolate per segment.
        beta: (n_ts, 4) tensor of four angles (see turtle_shape_hinges).
        lengths: (n_ts, 9) tensor of the lengths of the segments (see turtle_shape_hinges for the ordering).
        close_gait: Bool, whether to close the gait by appending the first point at the end.
        return_discretized: Bool, whether to return the discretized spline or not.
    
    Returns:
        Tensor: (n_ts, n_pos, 3) tensor representing the positions of the swimmer over time.
        ws: (n_ts, 4) tensor representing the discretized angles of the swimmer over time.
    '''
    n_cp = params.shape[0]
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp + close_gait)
    
    all_cps = generate_control_points(params, close_gait=True)

    # interpolate spline
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, all_cps, close_spline=close_gait)
    spline = NaturalCubicSpline(spline_coeffs)
    alphas_mass_logits = spline.evaluate(t)
    # print(all_cps.shape)
    
    hinge_vertices = turtle_shape_hinges(alphas_mass_logits[:, :4], beta.reshape(1, -1), lengths)
    all_mass_logits = turtle_allocate_masses(alphas_mass_logits[:, 4:], n_pts_per_segment)
    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])
    if return_discretized:
        return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges), all_mass_logits, alphas_mass_logits
    else:
        return interpolate_hinges(hinge_vertices, n_pts_per_segment, edges), all_mass_logits
    
def turtle_allocate_masses(masses_per_segment, n_pts_per_segment):
    '''
    Generate a function that returns the anisotropy direction of the turtle at each point.

    Args:
        masses_per_segment: (?, 9) tensor of the masses of the segments.
        n_pts_per_segment: Int, number of points to interpolate per segment.
    
    Returns:
        masses: (?, (n_edges*n_pts_per_segment + n_used_edges)) tensor representing the masses of the turtle at each point.
    '''

    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])
    n_edges = edges.shape[0]
    n_used_hinges = torch.unique(edges).shape[0]
    id_pos_to_edge = torch.zeros(n_edges * n_pts_per_segment + n_used_hinges, dtype=torch.long)
    id_pos_to_edge[:n_edges * n_pts_per_segment] = torch.repeat_interleave(torch.arange(n_edges), n_pts_per_segment, dim=0)
    id_pos_to_edge[n_edges * n_pts_per_segment    ] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 1] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 2] = 1
    id_pos_to_edge[n_edges * n_pts_per_segment + 3] = 2
    id_pos_to_edge[n_edges * n_pts_per_segment + 4] = 3
    id_pos_to_edge[n_edges * n_pts_per_segment + 5] = 4
    id_pos_to_edge[n_edges * n_pts_per_segment + 6] = 5
    id_pos_to_edge[n_edges * n_pts_per_segment + 7] = 6
    id_pos_to_edge[n_edges * n_pts_per_segment + 8] = 7
    id_pos_to_edge[n_edges * n_pts_per_segment + 9] = 8
    return masses_per_segment[..., id_pos_to_edge]

def turtle_allocate_arm_logits_to_segments(logits_per_arm):
    '''
    Args:
        params (torch.tensor of shape (n, 4 + 9)): the reshaped optimization parameters.
        
    Returns:
        logits_per_segments (torch.tensor of shape (n_cp, 9)): Logits for the points.
    '''
    logits_per_segments = torch.zeros(size=(logits_per_arm.shape[0], 9)) # cannot lift the trunk
    logits_per_segments[:, [1, 5]] = logits_per_arm[:, 0].unsqueeze(1)
    logits_per_segments[:, [2, 6]] = logits_per_arm[:, 1].unsqueeze(1)
    logits_per_segments[:, [3, 7]] = logits_per_arm[:, 2].unsqueeze(1)
    logits_per_segments[:, [4, 8]] = logits_per_arm[:, 3].unsqueeze(1)
    return logits_per_segments

def turtle_generate_fun_anisotropy_dir(n_pts_per_segment):
    '''
    Generate a function that returns the anisotropy direction of the turtle at each point.

    Args:
        n_pts_per_segment: Int, number of points to interpolate per segment.
    
    Returns:
        Function: A function that returns the anisotropy direction of the turtle at each point.
    '''

    edges = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])
    n_edges = edges.shape[0]
    n_used_hinges = torch.unique(edges).shape[0]
    offset_edges = edges + n_edges * n_pts_per_segment
    id_pos_to_edge = torch.zeros(n_edges * n_pts_per_segment + n_used_hinges, dtype=torch.long)
    id_pos_to_edge[:n_edges * n_pts_per_segment] = torch.repeat_interleave(torch.arange(n_edges), n_pts_per_segment, dim=0)
    id_pos_to_edge[n_edges * n_pts_per_segment    ] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 1] = 0
    id_pos_to_edge[n_edges * n_pts_per_segment + 2] = 1
    id_pos_to_edge[n_edges * n_pts_per_segment + 3] = 2
    id_pos_to_edge[n_edges * n_pts_per_segment + 4] = 3
    id_pos_to_edge[n_edges * n_pts_per_segment + 5] = 4
    id_pos_to_edge[n_edges * n_pts_per_segment + 6] = 5
    id_pos_to_edge[n_edges * n_pts_per_segment + 7] = 6
    id_pos_to_edge[n_edges * n_pts_per_segment + 8] = 7
    id_pos_to_edge[n_edges * n_pts_per_segment + 9] = 8
    
    def fun_anisotropy_dir(pos):
        '''
        Args:
            hinge_vertices: (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) tensor representing the positions of the turtle which has been generated using the turtle_shape_hinges function.

        Returns:
            torch.Tensor of shape (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3) representing the interpolated anisotropy directions.
        '''
        dir_edge = pos[..., offset_edges[:, 1], :] - pos[..., offset_edges[:, 0], :]
        dir_edge = dir_edge / torch.norm(dir_edge, dim=-1, keepdim=True) # shape: (n_ts, n_edges, 3)
        interpolated_anisotropy_dirs = dir_edge[..., id_pos_to_edge, :] # shape: (n_ts, (n_edges*n_pts_per_segment + n_used_edges), 3)
        
        return interpolated_anisotropy_dirs

    return fun_anisotropy_dir
