import numpy as np
from physics_quantities import momentum_dot, dot_momentum_
from scipy.linalg import lstsq
import torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

def compute_last_adjoint(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, ge, grad_obj):
    '''
    Compute the previous adjoint vector at time T

    Args:
        Ps: (n_points, 3) array the start positions of the system (including rigid transformation), time T-1
        Pe: (n_points, 3) array the end positions of the system (including rigid transformation), time T
        Ts: (n_points, 3) array the start tangents of the system (including rigid transformation), time T-1
        Te: (n_points, 3) array the end tangents of the system (including rigid transformation), time T
        a_weight: anisotropy of local dissipations metrics in the direction of the tangent
        b_weight: b_weight for anisotropy of local dissipations metrics
        ge: (7,) array representing a rigid transformation at the end of the time interval, time T
        grad_obj: (7,) gradient of the objective with respect to the final rigid transformation, time T

    Returns:
        we: (6,) the adjoint vector at the end of the time interval
    '''
    dμ_dge = np.array([momentum_dot(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, ge, ei, wrt_end=True) for ei in np.identity(7)]).T
    we = lstsq(dμ_dge.T, - grad_obj)[0]
    return we

def compute_last_gradient_pos_(Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, we, fun_anisotropy_dir):
    '''
    Compute the gradient of the objective wrt the shape at time T

    Args:
        Same as above
        Ps_: (n_points, 3) array the start positions of the system (excluding rigid transformation), time T-1
        Pe_: (n_points, 3) array the end positions of the system (excluding rigid transformation), time T
        gs: (7,) array representing a rigid transformation at the start of the time interval, time T-1
        ge: (7,) array representing a rigid transformation at the end of the time interval, time T
        we: (6,) the adjoint vector at the start of the time interval, time T
        fun_anisotropy_dir: function that returns the tangents/normals of the system at time 0

    Returns:
        dJ_dPe_: (n_points, 3) array the gradient of the cost wrt the untransformed positions, time T
    '''
    return (
        dot_momentum_(Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, we, fun_anisotropy_dir, wrt_end=True)
    )
    
def compute_first_gradient_pos_(Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, we, fun_anisotropy_dir):
    '''
    Compute the gradient of the objective wrt the shape at time 0

    Args:
        Same as above

    Returns:
        dJ_dPs_: (n_points, 3) array the gradient of the cost wrt the untransformed positions, time 0
    '''
    return (
        dot_momentum_(Ps_, Pe_, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, gs, ge, we, fun_anisotropy_dir, wrt_end=False)
    )

def step_backward_adjoint(Ps, Pe, Pe_next, Ts, Te, Te_next, Ms, Me, Me_next, a_weight_s, a_weight_e, a_weight_e_next, b_weight_s, b_weight_e, b_weight_e_next, ge, we, grad_obj):
    '''
    Compute the previous adjoint vector at time t

    Args:
        Same as above, except that positions and tangents are now transformed
        Pe_next: (n_points, 3) array the end positions of the system (including rigid transformation), time t+1
        Te_next: (n_points, 3) array the end tangents of the system (including rigid transformation), time t+1
        Me_next: (n_points, 1) array the masses time t+1
        a_weight_e_next: (n_points, 1) array anisotropy of local dissipations metrics in the direction of the tangent, time t+1
        b_weight_e_next: (n_points, 1) array b_weight for anisotropy of local dissipations metrics, time t+1
        we: (6,) the adjoint vector at the end of the time interval, time t+1
        grad_obj: (7,) gradient of the objective with respect to the rigid transformation, time t

    Returns:
        ws: (6,) the adjoint vector at the start of the time interval, time t
    '''

    dμ_dge = np.array([momentum_dot(Ps, Pe, Ts, Te, Ms, Me, a_weight_s, a_weight_e, b_weight_s, b_weight_e, ge, ei, wrt_end=True) for ei in np.identity(7)]).T
    dμ_dgs_next = np.array([momentum_dot(Pe, Pe_next, Te, Te_next, Me, Me_next, a_weight_e, a_weight_e_next, b_weight_e, b_weight_e_next, ge, ei, wrt_end=False) for ei in np.identity(7)]).T
    
    ws = lstsq(dμ_dge.T, - dμ_dgs_next.T @ we - grad_obj)[0]
    
    return ws

def step_backward_grad_pos_(Ps_, Pe_, Pe_next_, Ms, Me, Me_next, a_weight_s, a_weight_e, a_weight_e_next, b_weight_s, b_weight_e, b_weight_e_next, gs, ge, ge_next, we, we_next, fun_anisotropy_dir):
    '''
    Compute the gradient at time t

    Args:
        Same as above, except that positions and tangents are now transformed
        gs: (7,) array representing a rigid transformation at the start of the time interval, time t-1
        we: (6,) the adjoint vector at the start of the time interval, time t
        we_next: (6,) the adjoint vector at the end of the time interval, time t+1

    Returns:
        dJ_dPe_: (n_points, 3) array the gradient of the cost wrt the untransformed positions, time t
    '''
    return (
        dot_momentum_(Pe_, Pe_next_, Me, Me_next, a_weight_s, a_weight_e, b_weight_s, b_weight_e, ge, ge_next, we_next, fun_anisotropy_dir, wrt_end=False) +
        dot_momentum_(Ps_, Pe_, Ms, Me, a_weight_e, a_weight_e_next, b_weight_e, b_weight_e_next, gs, ge, we, fun_anisotropy_dir, wrt_end=True)
    )

def multiple_steps_backward_pos_(pos_, pos, tangents, masses, a_weights, b_weights, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir):
    '''
    Use step_backward multiple times and returns the gradient of the objective with respect to the shapes.
    Note that this assumes the dissipation parameters to be constant through time.
    
    Args:
        Same as above
        pos_: (n_steps, n_points, 3) the positions of the non-transformed points
        pos: (n_steps, n_points, 3) the positions of the rigidly transformed points
        tangents: (n_steps, n_points, 3) the tangents of the rigidly transformed points
        masses: (n_steps, n_points) the mass allocated to each point through time
        a_weights: (n_steps, n_points) array representing the a parameters for anisotropy of local dissipations metrics
        b_weights: (n_steps, n_points) array representing the b parameters for anisotropy of local dissipations metrics
        g: (n_steps, 7) the optimal rigid transformations
        grad_obj_g: the gradient of the objective with respect to the rigid transformations, ∂J/∂P.∂P/∂g or ∂J/∂g if J is directly a function of the rigid transformation, shape (n_steps, 7)
        grad_obj_pos_: the partial gradient of the objective with respect to the unregistered positions, ∂J/∂P.∂P/∂P_, shape (n_steps, 3*n_points) flattened from (n_steps, n_points, 3)
        
    Returns:
        grad_shape_obj: (n_step, 3*n_points) the gradient of the objective with respect to the non rigidly transformed points
    '''
    n_steps = pos.shape[0]
    
    # Compute the last adjoint
    adjoints = np.zeros(shape=(n_steps, 6))
    adjoints[-1] = compute_last_adjoint(
        pos[-2], pos[-1], tangents[-2], tangents[-1], 
        masses[-2].reshape(-1, 1), masses[-1].reshape(-1, 1), 
        a_weights[-2].reshape(-1, 1), a_weights[-1].reshape(-1, 1), 
        b_weights[-2].reshape(-1, 1), b_weights[-1].reshape(-1, 1), 
        g[-1], grad_obj_g[-1]
    )
    
    # Compute the gradient wrt shape
    grad_shape_obj = np.zeros(shape=(n_steps, 3 * pos.shape[1]))
    grad_shape_obj[-1] = compute_last_gradient_pos_(
        pos_[-2], pos_[-1], 
        masses[-2].reshape(-1, 1), masses[-1].reshape(-1, 1), 
        a_weights[-2].reshape(-1, 1), a_weights[-1].reshape(-1, 1), 
        b_weights[-2].reshape(-1, 1), b_weights[-2].reshape(-1, 1),
        g[-2], g[-1], adjoints[-1], fun_anisotropy_dir,
    ).reshape(-1,)
    
    # Loop over time
    for step in np.arange(1, n_steps-1)[::-1]:
        
        adjoints[step] = step_backward_adjoint(
            pos[step-1], pos[step], pos[step+1], 
            tangents[step-1], tangents[step], tangents[step+1],
            masses[step-1].reshape(-1, 1), masses[step].reshape(-1, 1), masses[step+1].reshape(-1, 1),
            a_weights[step-1].reshape(-1, 1), a_weights[step].reshape(-1, 1), a_weights[step+1].reshape(-1, 1), 
            b_weights[step-1].reshape(-1, 1), b_weights[step].reshape(-1, 1), b_weights[step+1].reshape(-1, 1), 
            g[step], adjoints[step+1], grad_obj_g[step]
        )

        grad_shape_obj[step] = step_backward_grad_pos_(
            pos_[step-1], pos_[step], pos_[step+1], 
            masses[step-1].reshape(-1, 1), masses[step].reshape(-1, 1), masses[step+1].reshape(-1, 1),
            a_weights[step-1].reshape(-1, 1), a_weights[step].reshape(-1, 1), a_weights[step+1].reshape(-1, 1), 
            b_weights[step-1].reshape(-1, 1), b_weights[step].reshape(-1, 1), b_weights[step+1].reshape(-1, 1), 
            g[step-1], g[step], g[step+1], adjoints[step], adjoints[step+1], fun_anisotropy_dir,
        ).reshape(-1,)
        
    grad_shape_obj[0] = compute_first_gradient_pos_(
        pos_[0], pos_[1], 
        masses[0].reshape(-1, 1), masses[1].reshape(-1, 1), 
        a_weights[0].reshape(-1, 1), a_weights[1].reshape(-1, 1), 
        b_weights[0].reshape(-1, 1), b_weights[1].reshape(-1, 1), 
        g[0], g[1], adjoints[1], fun_anisotropy_dir,
    ).reshape(-1,)

    grad_shape_obj = grad_shape_obj + grad_obj_pos_

    return grad_shape_obj
