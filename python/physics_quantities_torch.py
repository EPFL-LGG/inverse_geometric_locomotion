import torch
from utils import dot_vec_torch, qmultiply_torch, qinv_torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

def P_dot_torch(g, P, g_dot):
    '''Compute the sensitivities of the positions P wrt g
    
    Args:
        g: (7,) tensor representing a rigid transformation
        P: (N, 3) tensor the points the transformed points using g
        g_dot: (7,) tensor representing the time derivative of the rigid transformation
        
    Returns:
        (N, 3) tensor the sensitivities of the points
    '''
    q_dot = g_dot[:4]
    b_dot = g_dot[4:]
    
    q = g[:4]
    ω_dot = 2.0 * qmultiply_torch(q_dot, qinv_torch(q))[:3].unsqueeze(0)
    
    Pdot = torch.cross(ω_dot, P, dim=-1) + b_dot
    
    return Pdot

def energy_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e):
    '''Compute the energy between time t-1 and t acting on the shape given the shape change
    
    Args:
        Ps: (N, 3) torch tensor of current positions of the system (including rigid transformation)
        Pe: (N, 3) torch tensor of next positions of the system (including rigid transformation)
        Ts: (N, 3) torch tensor of current tangents of the system (including rigid transformation)
        Te: (N, 3) torch tensor of next tangents of the system (including rigid transformation)
        Ms: (N, 1) torch tensor of masses of the current prim gamma_curr
        Me: (N, 1) torch tensor of masses of the next prim gamma_next
        a_weight_*: scalar or (N, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (N, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        
    Returns:
        (3,) torch tensor of the force acting on the shape
    '''
    ΔP = Pe - Ps
    ΔP2 = dot_vec_torch(ΔP, ΔP)
    e1 = torch.sum(Ms * (a_weight_s * ΔP2 + b_weight_s * dot_vec_torch(ΔP, Ts) ** 2))
    e2 = torch.sum(Me * (a_weight_e * ΔP2 + b_weight_e * dot_vec_torch(ΔP, Te) ** 2))
    
    return (e1 + e2) / 2.0

vmap_energy_torch = torch.vmap(energy_torch, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

def force_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e):
    '''Compute the force acting on the shape given the shape change
    
    Args:
        Ps: (N, 3) torch tensor of current positions of the system (including rigid transformation)
        Pe: (N, 3) torch tensor of next positions of the system (including rigid transformation)
        Ts: (N, 3) torch tensor of current tangents of the system (including rigid transformation)
        Te: (N, 3) torch tensor of next tangents of the system (including rigid transformation)
        Ms: (N, 1) torch tensor of masses of the current prim gamma_curr
        Me: (N, 1) torch tensor of masses of the next prim gamma_next
        a_weight_*: scalar or (N, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (N, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        
    Returns:
        (3,) torch tensor of the force acting on the shape
    '''
    ΔP = Pe - Ps
    F1 = a_weight_s * ΔP + b_weight_s * dot_vec_torch(ΔP, Ts) * Ts
    # F1 = - (a_weight_s + b_weight_s) * ΔP + b_weight_s * dot_vec_torch(ΔP, Ts) * Ts
    F1 = - Ms * F1

    F2 = a_weight_e * ΔP + b_weight_e * dot_vec_torch(ΔP, Te) * Te
    # F2 = - (a_weight_e + b_weight_e) * ΔP + b_weight_e * dot_vec_torch(ΔP, Te) * Te
    F2 = - Me * F2
    
    return torch.sum(F1 + F2, dim=0)

def torque_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e):
    '''Compute the torque acting on the shape given the shape change
    
    Args:
        Same as force
        
    Returns:
        (3,) torch tensor of the torque acting on the shape
    '''
    ΔP = Pe - Ps
    T1 = a_weight_s * ΔP + b_weight_s * dot_vec_torch(ΔP, Ts) * Ts
    # T1 = (a_weight_s + b_weight_s) * ΔP - b_weight_s * dot_vec_torch(ΔP, Ts) * Ts
    T1 = Ms * torch.cross(T1, Pe, dim=-1)
    
    T2 = a_weight_e * ΔP + b_weight_e * dot_vec_torch(ΔP, Te) * Te
    # T2 = (a_weight_e + b_weight_e) * ΔP - b_weight_e * dot_vec_torch(ΔP, Te) * Te
    T2 = Me * torch.cross(T2, Ps, dim=-1)
    
    return torch.sum(T1 + T2, dim=0)

def momentum_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, force_prev, torque_prev):
    '''Compute the Δforce and Δtorque acting on the shape given the shape change
    
    Args:
        Same as force and torque
        force_prev: (3, 1) torch tensor of the Δforce acting on the shape in the previous iteration
        torque_prev: (3, 1) torch tensor of the Δtorque acting on the shape in the previous iteration
        
    Returns:
        (6,) torch tensor of the Δforce and Δtorque acting on the shape
    '''
    force_ = force_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e) - force_prev 
    torque_ = torque_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e) - torque_prev
    return torch.cat((force_, torque_), dim=0)
