import numpy as np
import torch

from physics_quantities_torch import momentum_torch
from scipy.spatial.transform import Rotation as R
from utils import (dot_vec, euc_transform_torch, qmultiply, qinv)

########################    
##### Base Quantities
########################

def energy(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e):
    '''Compute the energy between time t-1 and t acting on the shape given the shape change
    
    Args:
        Ps: (N, 3) array of current positions of the system (including rigid transformation)
        Pe: (N, 3) array of next positions of the system (including rigid transformation)
        Ts: (N, 3) array of current tangents of the system (including rigid transformation)
        Te: (N, 3) array of next tangents of the system (including rigid transformation)
        Ms: (N, 1) array of masses of the current prim gamma_curr
        Me: (N, 1) array of masses of the next prim gamma_next
        a_weight_*: scalar or (N, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (N, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        
    Returns:
        (3,) array of the force acting on the shape
    '''
    ΔP = Pe - Ps
    ΔP2 = dot_vec(ΔP, ΔP)
    e1 = np.sum(Ms * (a_weight_s * ΔP2 + b_weight_s * dot_vec(ΔP, Ts) ** 2))
    e2 = np.sum(Me * (a_weight_e * ΔP2 + b_weight_e * dot_vec(ΔP, Te) ** 2))
    
    return (e1 + e2) / 2.0

def force(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e):
    '''Compute the force between time t-1 and t acting on the shape given the shape change
    
    Args:
        Ps: (N, 3) array of current positions of the system (including rigid transformation)
        Pe: (N, 3) array of next positions of the system (including rigid transformation)
        Ts: (N, 3) array of current tangents of the system (including rigid transformation)
        Te: (N, 3) array of next tangents of the system (including rigid transformation)
        Ms: (N, 1) array of masses of the current prim gamma_curr
        Me: (N, 1) array of masses of the next prim gamma_next
        a_weight_*: scalar or (N, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (N, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        
    Returns:
        (3,) array of the force acting on the shape
    '''
    ΔP = Pe - Ps
    F1 = a_weight_s * ΔP + b_weight_s * dot_vec(ΔP, Ts) * Ts
    F1 = - Ms * F1

    F2 = a_weight_e * ΔP + b_weight_e * dot_vec(ΔP, Te) * Te
    F2 = - Me * F2
    
    return np.sum(F1 + F2, axis=0)

def torque(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e):
    '''Compute the torque acting on the shape given the shape change
    
    Args:
        Same as force
        
    Returns:
        (3,) array of the torque acting on the shape
    '''
    ΔP = Pe - Ps
    T1 = a_weight_s * ΔP + b_weight_s * dot_vec(ΔP, Ts) * Ts
    T1 = Ms * np.cross(T1, Pe, axis=-1)
    
    T2 = a_weight_e * ΔP + b_weight_e * dot_vec(ΔP, Te) * Te
    T2 = Me * np.cross(T2, Ps, axis=-1)
    
    return np.sum(T1 + T2, axis=0)

def momentum(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, force_prev, torque_prev):
    '''Compute the Δforce and Δtorque acting on the shape given the shape change
    
    Args:
        Same as force and mtorque
        force_prev: (3, 1) array of the Δforce acting on the shape in the previous iteration
        torque_prev: (3, 1) array of the Δtorque acting on the shape in the previous iteration
        
    Returns:
        (6,) array of the Δforce and Δtorque acting on the shape
    '''
    force_ = force(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e) - force_prev 
    torque_ = torque(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e) - torque_prev
    return np.concatenate((force_, torque_), axis=0)

############################    
##### First-Order Quantities
############################

def P_dot(g, P, g_dot):
    '''Compute the sensitivities of the positions P wrt g
    
    Args:
        g: (7,) array representing a rigid transformation
        P: (N, 3) array the points the transformed points using g
        g_dot: (7,) array representing the time derivative of the rigid transformation
        
    Returns:
        (N, 3) array the sensitivities of the points
    '''
    q_dot = g_dot[:4]
    b_dot = g_dot[4:]
    
    q = g[:4]
    b = g[4:].reshape(1, 3)
    ω_dot = 2.0 * qmultiply(q_dot, qinv(q))[:3]
    
    Pdot = np.cross(ω_dot, P - b, axis=-1) + b_dot
    
    return Pdot

def dot_P(g, P_, P_dot):
    '''Compute the pullback of P_dot wrt g: P_dot^T.∂P/∂g
    
    Args:
        g: (7,) array representing a rigid transformation
        P: (N, 3) array the points the transformed points using g
        P_dot: (7,) array representing the time derivative of the rigid transformation
        
    Returns:
        (N, 3) array the sensitivities of the points
    '''
    
    P_torch_ = torch.tensor(P_)
    P_torch_.requires_grad = True
    P = euc_transform_torch(torch.tensor(g), P_torch_)
    P.backward(P_dot)
    
    return P_torch_.grad.numpy()

def P0_dot(g, P0, P0_dot):
    '''Compute the sensitivities of the positions P wrt P_0
    
    Args:
        g: (7,) array representing a rigid transformation
        P0: (N, 3) array the points to be transformed
        P0_dot: (N, 3) array the change in the points to be transformed
        
    Returns:
        (N, 3) array the sensitivities of the points
    '''
    rot = R.from_quat(g[:4])
    return rot.apply(P0_dot)

def dot_P0(g, P, P_dot):
    '''Compute the pullback of P_dot wrt P0
    
    Args:
        g: (7,) array representing a rigid transformation
        P0: (N, 3) array the points to be transformed
        P_dot: (N, 3) array the vector field to be pulled
        
    Returns:
        (N, 3) array the pullback of the points
    '''
    rot = R.from_quat(qinv(g[:4])) # rot.T
    return rot.apply(P_dot)
    
def T_dot(g, T, g_dot):
    '''Compute the sensitivities of the tangents T wrt g
    
    Args:
        Same as P_dot
        T: (N, 3) array the tangents to be transformed
        
    Returns:
        (N, 3) array the sensitivities of the tangents
    '''
    q_dot = g_dot[:4]
    
    q = g[:4]
    ω_dot = 2.0 * qmultiply(q_dot, qinv(q))[:3]
    
    Tdot = np.cross(ω_dot, T, axis=-1)
    
    return Tdot

def T0_dot(g, T0, T0_dot):
    '''Compute the sensitivities of the positions P wrt P_0
    
    Args:
        g: (7,) array representing a rigid transformation
        T0: (N, 3) array the tangents to be transformed
        T0_dot: (N, 3) array the change in the tangents to be transformed
        
    Returns:
        (N, 3) array the sensitivities of the tangents
    '''
    rot = R.from_quat(g[:4])
    return rot.apply(T0_dot)

def force_dot(Ps, Pe, Ts, Te, Ms, Me, P_dot, T_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, wrt_end=True):
    '''Compute the sensitivities of the force acting on the shape given the shape change
    
    Args:
        Same as force and torque
        P_dot: (N, 3) array of the changes of the positions of the system
        T_dot: (N, 3) array of the changes of the tangents of the system
        wrt_end: whether P_dot and T_dot represent changes at the begining or at the end of the time interval
        
    Returns:
        (3,) array of the time derivative of the force acting on the shape
    '''
    
    ΔP = Pe - Ps
    
    if wrt_end:
        F1_dot = a_weight_s * P_dot + b_weight_s * dot_vec(P_dot, Ts) * Ts
        F2_dot = a_weight_e * P_dot + b_weight_e * dot_vec(P_dot, Te) * Te
        F2_dot += b_weight_e * (dot_vec(ΔP, T_dot) * Te + dot_vec(ΔP, Te) * T_dot)
    else:
        F1_dot = - a_weight_s * P_dot - b_weight_s * dot_vec(P_dot, Ts) * Ts
        F1_dot += b_weight_s * (dot_vec(ΔP, T_dot) * Ts + dot_vec(ΔP, Ts) * T_dot)
        F2_dot = - a_weight_e * P_dot - b_weight_e * dot_vec(P_dot, Te) * Te
    F1_dot = - Ms * F1_dot
    F2_dot = - Me * F2_dot
    
    return np.sum(F1_dot + F2_dot, axis=0)
    
        
def torque_dot(Ps, Pe, Ts, Te, Ms, Me, P_dot, T_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, wrt_end=True):
    '''Compute the sensitivities of the torque acting on the shape given the shape change
    
    Args:
        Same as force and torque
        P_dot: (N, 3) array of the changes of the positions of the system
        T_dot: (N, 3) array of the changes of the tangents of the system
        wrt_end: whether P_dot and T_dot represent changes at the begining or at the end of the time interval
        
    Returns:
        (3,) array of the time derivative of the torque acting on the shape
    '''
    ΔP = Pe - Ps
    
    if wrt_end:
        t1 = a_weight_s * ΔP + b_weight_s * dot_vec(ΔP, Ts) * Ts
        t1_dot = a_weight_s * P_dot + b_weight_s * dot_vec(P_dot, Ts) * Ts
        T1_dot = Ms * (np.cross(t1, P_dot, axis=-1) + np.cross(t1_dot, Pe, axis=-1))
        
        T2_dot = a_weight_e * P_dot + b_weight_e * dot_vec(P_dot, Te) * Te
        T2_dot += b_weight_e * (dot_vec(ΔP, T_dot) * Te + dot_vec(ΔP, Te) * T_dot)
        T2_dot = Me * np.cross(T2_dot, Ps, axis=-1)
    else:
        T1_dot = - a_weight_s * P_dot - b_weight_s * dot_vec(P_dot, Ts) * Ts
        T1_dot += b_weight_s * (dot_vec(ΔP, T_dot) * Ts + dot_vec(ΔP, Ts) * T_dot)
        T1_dot = Ms * np.cross(T1_dot, Pe, axis=-1)

        t2 = a_weight_e * ΔP + b_weight_e * dot_vec(ΔP, Te) * Te
        t2_dot = - a_weight_e * P_dot - b_weight_e * dot_vec(P_dot, Te) * Te
        T2_dot = Me * (np.cross(t2, P_dot, axis=-1) + np.cross(t2_dot, Ps, axis=-1))

    return np.sum(T1_dot + T2_dot, axis=0)

def momentum_dot(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, g, g_dot, wrt_end=True):
    '''Compute the JVP of the momentum at time t given the positioning change: ∂μ/∂g.g_dot at time t-1 or t
    
    In particular, we interpret g and g_dot as the rigid transformation at time t-1 or t-1 and the
    inifinitesimal change in the rigid transformation at time t-1 or t. The momentum is always evaluated
    in the time interval t-1 to t.
    
    Args:
        Same as force and torque
        g: (7,) array representing a rigid transformation
        g_dot: (7,) array representing the infinitesimal change of the rigid transformation
        wrt_end: whether g and g_dot represent the positioning at the begining or at the end of the time interval
        
    Returns:
        (6,) array of the JVP of the momentum
    '''
    if wrt_end:
        Pse_dot = P_dot(g, Pe, g_dot)
        Tse_dot = T_dot(g, Te, g_dot)
    else:
        Pse_dot = P_dot(g, Ps, g_dot)
        Tse_dot = T_dot(g, Ts, g_dot)
    
    force_dot_ = force_dot(Ps, Pe, Ts, Te, Ms, Me, Pse_dot, Tse_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, wrt_end=wrt_end)
    torque_dot_ = torque_dot(Ps, Pe, Ts, Te, Ms, Me, Pse_dot, Tse_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, wrt_end=wrt_end)
    
    return np.concatenate((force_dot_ , torque_dot_), axis=0)  

def momentum_dot_(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, g, Pe0_dot, Te0_dot, wrt_end=True):
    '''Compute the JVP of the momentum at time t given the shape variation at time t-1 or t'''
    Pe_dot = P0_dot(g, None, Pe0_dot)
    Te_dot = T0_dot(g, None, Te0_dot)
    
    force_dot_ = force_dot(Ps, Pe, Ts, Te, Ms, Me, Pe_dot, Te_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, wrt_end=wrt_end)
    torque_dot_ = torque_dot(Ps, Pe, Ts, Te, Ms, Me, Pe_dot, Te_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, wrt_end=wrt_end)
    
    return np.concatenate((force_dot_ , torque_dot_), axis=0)  

def dot_momentum_(Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, δμ, fun_anisotropy_dir, wrt_end=True):
    '''
    Compute the VJP of a momentum change (δμ^T.dμ/dP_)^T with respect to the reference positions.
    The momentum is evaluated at time t, and differentiated against the position at time t-1 or t depending on wrt_end.
    '''
    
    Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, δμ = map(torch.tensor, (Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, δμ))
    if wrt_end: 
        Pe_.requires_grad = True
    else:
        Ps_.requires_grad = True
        
    Ps = euc_transform_torch(gs, Ps_)
    Pe = euc_transform_torch(ge, Pe_)
        
    Ts = fun_anisotropy_dir(Ps)
    Te = fun_anisotropy_dir(Pe)
    
    dot_momentum = δμ @ momentum_torch(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, torch.zeros(size=(3,)), torch.zeros(size=(3,)))
    dot_momentum.backward(torch.ones_like(dot_momentum))
    
    if wrt_end: 
        return Pe_.grad.numpy()
    else:
        return Ps_.grad.numpy()
