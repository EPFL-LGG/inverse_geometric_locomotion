import os
import sys as _sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CUBICSPLINES = os.path.join(os.path.dirname(SCRIPT_PATH), 'ext/torchcubicspline')
_sys.path.append(PATH_TO_CUBICSPLINES)

import torch
from torchcubicspline import (
    natural_cubic_spline_coeffs, NaturalCubicSpline
)
from utils import register_points_torch, vmap_rotate_about_axis, register_points_torch_2d

def sine_generation(
    length_snake, width_snake, init_offset, shape_period, n_cycles, example_pos_,
    n_ts, n_s,
):
    '''Generate the positions of a sinusoidalsnake evolving in time
    
    Args:
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        n_s: int representing the number of points in the snake
        
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''
    ts = init_offset + n_cycles * torch.linspace(0.0, 1.0, n_ts).reshape(-1, 1) + shape_period * torch.linspace(0.0, 1.0, n_s).reshape(1, -1)
    # No rigid transformation
    pos_ = torch.zeros_like(example_pos_)
    pos_[..., 0] = length_snake * torch.linspace(0.0, 1.0, n_s) - length_snake / 2.0
    pos_[..., 1] = width_snake * torch.cos(2 * torch.pi * ts)
    
    return pos_

vmap_sine_generation = torch.vmap(sine_generation, in_dims=(0, 0, 0, 0, 0, 0, None, None))

def serpenoid_generation(
    sigma, center, theta, larger_axis, snake_length, wavelength, final_t,
    example_pos_, n_ts, n_s, flip_snake=False,
):
    '''Generate a serpenoid curve evolving in time
    
    Args:
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        n_s: int representing the number of points in the snake
        flip_snake: bool representing whether the snake should be flipped or not
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    t = final_t * torch.linspace(0.0, 1.0, n_ts)
    s = torch.linspace(0.0, 1.0, n_s-1)

    rot_theta = torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1),
        torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    ], dim=-1)
    # rot_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    smaller_axis = sigma * larger_axis
    ws = torch.stack([
        larger_axis * torch.cos(2.0 * torch.pi * t),
        smaller_axis * torch.sin(2.0 * torch.pi * t)
    ], dim=-1)
    ws = torch.matmul(ws, rot_theta.swapaxes(-2, -1)) + center.unsqueeze(-2)

    spatial_oscillation_phis = torch.stack([
        -(torch.cos(2.0 * torch.pi * s / wavelength) - 1.0),
        torch.sin(2.0 * torch.pi * s / wavelength),
    ], dim=-1)
    phis = torch.sum(ws.reshape(-1, 1, 2) * spatial_oscillation_phis.reshape(1, -1, 2), dim=2) * wavelength / (2.0 * torch.pi)
    
    integrand = torch.stack([
        torch.cos(phis) * snake_length / n_s,
        torch.sin(phis) * snake_length / n_s
    ], dim=-1)

    pos_ = torch.cumsum(torch.cat([torch.zeros_like(integrand)[..., 0, :].unsqueeze(-2), integrand], dim=1), dim=1)
    
    registration_points = torch.zeros_like(example_pos_[..., :2])
    registration_points[..., 0] = (1.0 - 2.0 * flip_snake) * 2.0 * torch.linspace(0.0, 1.0, n_s) - 1.0

    pos2d_reg_ = register_points_torch(pos_, registration_points)
    pos3d_reg_ = torch.zeros_like(example_pos_)
    pos3d_reg_[..., :2] = pos2d_reg_
    
    return pos3d_reg_

vmap_serpenoid_generation = torch.vmap(serpenoid_generation, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, None, None))

def serpenoid_generation_cubic_splines_varying_wavelengths(
    control_points, snake_length,
    example_pos_, n_ts, n_s, n_cp, close_snake=False, flip_snake=False,
):
    '''Generate a serpenoid curve evolving in time
    
    Args:
        control_points: (n_cp, 3) tensor representing the control points of the cubic spline, the first two dimensions are the serpenoid shape space's w1 and w2 and the third dimension is the wavelength
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        wavelengths: (n_ts,) tensor representing the wavelengths of the serpenoid
        n_ts: int representing the number of time steps
        n_s: int representing the number of points in the snake
        n_cp: int representing the number of control points
        flip_snake: bool representing whether the snake should be flipped or not
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    t = torch.linspace(0.0, 1.0, n_ts)
    s = torch.linspace(0.0, 1.0, n_s-1)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_snake)

    if close_snake:
        control_points = torch.cat([control_points, control_points[0].reshape(1, -1)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points)
    spline = NaturalCubicSpline(spline_coeffs)
    ws = spline.evaluate(t) # (n_ts, 3), in the [w1, w2] space + wavelength

    spatial_oscillation_phis = torch.stack([
        - torch.cos(2.0 * torch.pi * s.reshape(1, -1) / ws[:, 2].reshape(-1, 1)),
        torch.sin(2.0 * torch.pi * s.reshape(1, -1) / ws[:, 2].reshape(-1, 1)),
    ], dim=-1)
    phis = torch.sum(ws[:, :2].reshape(-1, 1, 2) * spatial_oscillation_phis, dim=2) * ws[:, 2].reshape(-1, 1) / (2.0 * torch.pi)
    phis_centered = phis - phis[:,0].reshape(-1, 1)
    
    integrand = torch.stack([
        torch.cos(phis_centered) * snake_length / n_s,
        torch.sin(phis_centered) * snake_length / n_s
    ], dim=-1)

    pos_ = torch.cumsum(torch.cat([torch.zeros_like(integrand)[..., 0, :].unsqueeze(-2), integrand], dim=1), dim=1)
    pos3d_ = torch.zeros_like(example_pos_)
    
    torch.manual_seed(0)
    registration_points = torch.zeros(size=(n_s, 2))
    registration_points[:, 0] = (1.0 - 2.0 * flip_snake) * snake_length * (torch.linspace(0.0, 1.0, n_s) - 0.5)
    registration_points[:, 1] = 0.01 * snake_length * torch.randn(size=(n_s,))
    pos3d_[..., :2] = register_points_torch_2d(pos_, registration_points.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False)
    return pos3d_
    

def serpenoid_generation_cubic_splines(
    control_points, snake_length, wavelength,
    example_pos_, n_ts, n_s, n_cp, close_snake=False, flip_snake=False,
):
    '''Generate a serpenoid curve evolving in time
    
    Args:
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        n_s: int representing the number of points in the snake
        n_cp: int representing the number of control points
        flip_snake: bool representing whether the snake should be flipped or not
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''

    t = torch.linspace(0.0, 1.0, n_ts)
    s = torch.linspace(0.0, 1.0, n_s-1)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_snake)

    if close_snake:
        control_points = torch.cat([control_points, control_points[0].reshape(1, -1)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points)
    spline = NaturalCubicSpline(spline_coeffs)
    ws = spline.evaluate(t)

    spatial_oscillation_phis = torch.stack([
        - torch.cos(2.0 * torch.pi * s / wavelength),
        torch.sin(2.0 * torch.pi * s / wavelength),
    ], dim=-1)
    phis = torch.sum(ws.reshape(-1, 1, 2) * spatial_oscillation_phis.reshape(1, -1, 2), dim=2) * wavelength / (2.0 * torch.pi)
    
    integrand = torch.stack([
        torch.cos(phis) * snake_length / n_s,
        torch.sin(phis) * snake_length / n_s
    ], dim=-1)

    pos_ = torch.cumsum(torch.cat([torch.zeros_like(integrand)[..., 0, :].unsqueeze(-2), integrand], dim=1), dim=1)
    pos3d_ = torch.zeros_like(example_pos_)
    
    torch.manual_seed(0)
    registration_points = torch.zeros(size=(n_s, 2))
    registration_points[:, 0] = (1.0 - 2.0 * flip_snake) * snake_length * (torch.linspace(0.0, 1.0, n_s) - 0.5)
    registration_points[:, 1] = 0.01 * snake_length * torch.randn(size=(n_s,))
    pos3d_[..., :2] = register_points_torch_2d(pos_, registration_points.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False)
    return pos3d_

def snake_angles_generation_cubic_splines(
    control_points, snake_length, broken_joint_ids, broken_joint_angles,
    example_pos_, n_ts, n_cp, close_snake=False, flip_snake=False, return_discretized=False,
):
    '''Generate a snake parameterized by turning angles evolving in time
    
    Args:
        control_points: (n_cp, n_operational_angles) tensor representing the control points of the cubic spline, it
        snake_length: float representing the length of the snake
        broken_joint_ids: list of int representing the indices of the broken joints
        broken_joint_angles: (n_ts, n_broken_angles) tensor representing the turning angles of the broken joints
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        n_cp: int representing the number of control points
        flip_snake: bool representing whether the snake should be flipped or not
        return_discretized: Bool, whether to return the discretized spline or not.
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''
    
    n_s = control_points.shape[1] + len(broken_joint_ids) + 2
    n_angles = n_s - 2
    edge_length = snake_length / (n_s - 1)
    
    t = torch.linspace(0.0, 1.0, n_ts)
    s = torch.linspace(0.0, 1.0, n_s-1)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_snake)

    if close_snake:
        control_points = torch.cat([control_points, control_points[0].reshape(1, -1)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points)
    spline = NaturalCubicSpline(spline_coeffs)
    operational_angles = spline.evaluate(t) # shape (n_ts, n_operational_angles)
    
    operational_joint_ids = [i for i in range(n_angles) if i not in broken_joint_ids]
    all_angles = torch.zeros(size=(n_ts, n_angles))
    all_angles[:, operational_joint_ids] = operational_angles
    all_angles[:, broken_joint_ids] = broken_joint_angles
    
    pos_ = integrate_snake_angles_constant_edge_length(all_angles, edge_length, n_ts)
    pos3d_ = torch.zeros_like(example_pos_)
    
    torch.manual_seed(0)
    registration_points = torch.zeros(size=(n_s, 2))
    registration_points[:, 0] = (1.0 - 2.0 * flip_snake) * snake_length * (torch.linspace(0.0, 1.0, n_s) - 0.5)
    registration_points[:, 1] = 0.01 * snake_length * torch.randn(size=(n_s,))
    pos3d_[..., :2] = register_points_torch_2d(pos_, registration_points.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False)
    
    if return_discretized:
        return pos3d_, operational_angles
    else:
        return pos3d_
    
    
def integrate_snake_angles_constant_edge_length(
    all_angles, edge_length, n_ts,
):
    '''Integrate the turning angles of the snake to get the positions
    
    Args:
        all_angles: (n_ts, n_angles) tensor representing the turning angles of the snake
        edge_length: float representing the length of the edges of the snake
        n_ts: int representing the number of time steps
    
    Returns:
        (n_ts, n_angles+2, 2) tensor representing the positions of the snake
    '''
    
    n_ts, n_angles = all_angles.shape
    all_cumulated_angles = torch.cumsum(torch.cat([torch.zeros(size=(n_ts, 1)), all_angles], dim=-1), dim=-1)
    
    pos_ = torch.zeros(size=(n_ts, n_angles+2, 2))
    pos_[..., 1:, 0] = edge_length * torch.cumsum(torch.cos(all_cumulated_angles), dim=-1)
    pos_[..., 1:, 1] = edge_length * torch.cumsum(torch.sin(all_cumulated_angles), dim=-1)
    
    return pos_
