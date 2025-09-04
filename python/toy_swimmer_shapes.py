'''
We reproduce the toy swimmer example from the paper "Robotic swimming in curved space via geometric phase" by Li et al. (2022).
'''

import torch

def compute_spherical_swimmer_position(theta_hv, radius, delta_theta_h):
    '''
    Compute the position of the 4 masses of swimmer on the sphere.
    
    Args:
        theta_hv: (..., 2) tensor of the horizontal and vertical angles of the swimmer
        radius (float): radius of the sphere
        delta_theta_h (float): horizontal angle between the two masses on the same hemisphere
        
    Returns:
        masses_positions: (..., 4, 3) tensor of the positions of the masses of the swimmer
    '''
    
    masses_positions = torch.zeros(size=theta_hv.shape[:-1] + (4, 3))
    masses_positions[..., 0, :] = radius * torch.stack([
        torch.cos(theta_hv[..., 0]),
        torch.sin(theta_hv[..., 0]),
        torch.zeros_like(theta_hv[..., 0])
    ], dim=-1)
    
    masses_positions[..., 1, :] = radius * torch.stack([
        torch.cos(theta_hv[..., 0] + delta_theta_h),
        torch.sin(theta_hv[..., 0] + delta_theta_h),
        torch.zeros_like(theta_hv[..., 0])
    ], dim=-1)
    
    masses_positions[..., 2, :] = radius * torch.stack([
        torch.cos(theta_hv[..., 1]),
        torch.zeros_like(theta_hv[..., 1]),
        torch.sin(theta_hv[..., 1])
    ], dim=-1)
    
    masses_positions[..., 3, :] = radius * torch.stack([
        torch.cos(-theta_hv[..., 1]),
        torch.zeros_like(theta_hv[..., 1]),
        torch.sin(-theta_hv[..., 1])
    ], dim=-1)

    return masses_positions
