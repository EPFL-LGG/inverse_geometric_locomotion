import contextlib
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.transform import Rotation as R
import sys
import torch
import torch.nn.functional as F

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

def dot_vec(a, b):
    '''
    Args:
        a: (N, 3) array
        b: (N, 3) array
    
    Returns:
        (N, 1) array
    '''
    return np.sum(a * b, axis=1)[:, None]

def dot_vec_torch(a, b):
    '''
    Args:
        a: (N, 3) array
        b: (N, 3) array
    
    Returns:
        (N, 1) array
    '''
    return torch.sum(a * b, dim=1)[:, None]

def euc_transform(g, X):
    '''
    Args:
        g: (7,) array representing a rigid transformation (real part last)
        X: (N, 3) array the points to be transformed
        
    Returns:
        (N, 3) array the transformed points
    '''
    q = g[:4]
    b = g[4:]
    
    rot = R.from_quat(q)
    return rot.apply(X) + b

def euc_transform_torch(g, X):
    '''
    Args:
        g: (7,) array representing a rigid transformation (real part last)
        X: (N, 3) array the points to be transformed
        
    Returns:
        (N, 3) array the transformed points
    '''
    q = g[:4]
    b = g[4:]
    
    rot = quaternion_to_matrix_torch(q)
    return X @ rot.T + b

vmap_euc_transform_torch = torch.vmap(euc_transform_torch, in_dims=(0, 0))
    
def euc_transform_T(g, T):
    '''Same as euc_transform, but disregards the translation component'''
    q = g[:4]
    b = g[4:]
    
    rot = R.from_quat(q)
    return rot.apply(T)

def euc_transform_T_torch(g, X):
    '''
    Args:
        g: (7,) array representing a rigid transformation (real part last)
        X: (N, 3) array the points to be transformed
        
    Returns:
        (N, 3) array the transformed points
    '''
    q = g[:4]
    b = g[4:]
    
    rot = quaternion_to_matrix_torch(q)
    return X @ rot.T

vmap_euc_transform_T_torch = torch.vmap(euc_transform_T_torch, in_dims=(0, 0))

# QUATERNION FUNCTIONS

def qmultiply(q1, q2):
    '''Multiply two quaternions'''
    q1_s = q1[3]
    q2_s = q2[3]
    q1_v = q1[:3]
    q2_v = q2[:3]
    
    real = q1_s * q2_s - np.dot(q1_v, q2_v)
    imag = q1_s * q2_v + q2_s * q1_v + np.cross(q1_v, q2_v)
    return np.array([imag[0], imag[1], imag[2], real])

def qbar(q):
    '''Take the conjugate of a quaternion'''
    return np.array([-q[0], -q[1], -q[2], q[3]])
    
def qinv(q):
    '''Take the inverse of a quaternion'''
    if np.linalg.norm(q) == 0:
        print("Error: division by zero")
    return qbar(q) / np.dot(q, q)

def qmultiply_torch(q1, q2):
    '''Multiply two quaternions'''
    q1_s = q1[3]
    q2_s = q2[3]
    q1_v = q1[:3]
    q2_v = q2[:3]
    
    real = (q1_s * q2_s - torch.dot(q1_v, q2_v)).reshape(-1,)
    imag = q1_s * q2_v + q2_s * q1_v + torch.cross(q1_v, q2_v, dim=0)
    return torch.cat([imag, real], dim=0)

vmap_qmultiply_torch = torch.vmap(qmultiply_torch, in_dims=(0, 0))

def qbar_torch(q):
    '''Take the conjugate of a quaternion'''
    q_bar = q.clone()
    q_bar[:3] = - q[:3]
    return q_bar

vmap_qbar_torch = torch.vmap(qbar_torch, in_dims=(0,))

def qinv_torch(q):
    '''Take the inverse of a quaternion'''
    return qbar_torch(q) / torch.dot(q, q)

vmap_qinv_torch = torch.vmap(qinv_torch, in_dims=(0,))

def quaternion_to_matrix_torch(quaternions):
    '''
    Convert rotations given as quaternions to rotation matrices.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.quaternion_to_matrix

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    '''
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_axis_angle(quaternions):
    '''
    Convert rotations given as quaternions to axis/angle.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    '''
    norms = torch.linalg.norm(quaternions[..., :-1], dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., -1])
    angles = 2.0 * half_angles
    eps = 1.0e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48.0
    )
    return quaternions[..., :-1] / sin_half_angles_over_angles

def axis_angle_to_quaternion(axis_angle):
    '''
    Convert rotations given as axis/angle to quaternions.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_quaternion

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    '''
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1.0e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48.0
    )
    quaternions = torch.cat(
        [axis_angle * sin_half_angles_over_angles, torch.cos(half_angles)], dim=-1
    )
    return quaternions

def _sqrt_positive_part(x):
    '''
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    '''
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions):
    '''
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Source: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.standardize_quaternion

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    '''
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)

def matrix_to_quaternion_torch(matrix):
    '''
    Convert rotations given as rotation matrices to quaternions.

    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    '''
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return torch.roll(standardize_quaternion(out), shifts=-1, dims=-1)

def print_quaternion(q, precision=2, suppress_small=True):
    '''Print a quaternion (torch tensor of size (4,)) in a human-readable format'''
    axis_angle = quaternion_to_axis_angle(q)
    print("Axis angle: {} (angle value: {:.1f} deg)".format(np.array2string(axis_angle.numpy(), precision=precision, suppress_small=suppress_small), np.linalg.norm(axis_angle) * 180 / np.pi))

def register_points(points1, points2):
    '''
    The rotation aligns the principal axes of the point clouds
    It should be applied to points1 using R.apply(points1), or to points2 using R.apply(points2, inverse=True)
    '''
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)
    PR1 = points1 - centroid1  # points relative to centroid
    PR2 = points2 - centroid2  # points relative to centroid
    return R.align_vectors(PR2, PR1)[0]

def register_points_torch(points1, points2, allow_flip=True):
    '''Assumes that the points are (batch_size, n_pts, 3) tensors'''
    centroid1 = torch.mean(points1, dim=1, keepdim=True)
    centroid2 = torch.mean(points2, dim=1, keepdim=True)

    PR1 = points1 - centroid1  # points relative to centroid
    PR2 = points2 - centroid2  # points relative to centroid

    U, S, Vh = torch.linalg.svd(torch.einsum("tsi, tsj -> tij", PR1, PR2))
    if not allow_flip:
        unflip = torch.eye(3).reshape(1, 3, 3).repeat(U.shape[0], 1, 1)
        detU = torch.linalg.det(U.detach())
        detV = torch.linalg.det(Vh.detach())
        unflip[:, 2, 2] = torch.sign(detU * detV)
        rot = Vh.transpose(1, 2) @ unflip @ U.transpose(1, 2)
    else:
        rot = Vh.transpose(1, 2) @ U.transpose(1, 2)

    registered = centroid2 + torch.einsum("tij, taj -> tai", rot, PR1)
    return registered

def register_points_torch_2d(points1, points2, allow_flip=True, return_rotation=False):
    '''Assumes that the points are (batch_size, n_pts, 2) tensors'''
    centroid1 = torch.mean(points1, dim=1, keepdim=True)
    centroid2 = torch.mean(points2, dim=1, keepdim=True)

    PR1 = points1 - centroid1  # points relative to centroid
    PR2 = points2 - centroid2  # points relative to centroid

    U, S, Vh = torch.linalg.svd(torch.einsum("tsi, tsj -> tij", PR1, PR2))
    if not allow_flip:
        unflip = torch.eye(2).reshape(1, 2, 2).repeat(U.shape[0], 1, 1)
        detU = torch.linalg.det(U.detach())
        detV = torch.linalg.det(Vh.detach())
        unflip[:, 1, 1] = torch.sign(detU * detV)
        rot = Vh.transpose(1, 2) @ unflip @ U.transpose(1, 2)
    else:
        rot = Vh.transpose(1, 2) @ U.transpose(1, 2)

    registered = centroid2 + torch.einsum("tij, taj -> tai", rot, PR1)
    if return_rotation:
        return registered, rot
    else:
        return registered

def align_point_cloud(pos, allow_flip=True):
    '''Aligns point clouds to xyz axes
    
    Args:
        pos (torch.Tensor of shape (n_points, 3)): The points to align.
        allow_flip (bool): If True, allow finding reflections of the points.
        
    Returns:
        rotated_pos torch.Tensor of shape (n_points, 3): The aligned points.
    '''
    centered_pos = pos - torch.mean(pos, dim=0, keepdim=True)
    cov = torch.einsum("si, sj -> ij", centered_pos, centered_pos)
    L, Q = torch.linalg.eigh(cov)
    Q = Q[:, torch.argsort(L, descending=True)]
    if not allow_flip:
        d = torch.sign(torch.linalg.det(Q))
        Q[:, 0] = d * Q[:, 0]
    
    rotated_pos = torch.einsum("ij, aj -> ai", Q, centered_pos)
    return rotated_pos

def align_point_clouds(pos, allow_flip=True):
    '''Aligns point clouds to xyz axes
    
    Args:
        pos (torch.Tensor of shape (n_batch, n_points, 3)): The points to align.
        allow_flip (bool): If True, allow finding reflections of the points.
        
    Returns:
        rotated_pos torch.Tensor of shape (n_points, 3): The aligned points.
    '''
    centered_pos = pos - torch.mean(pos, dim=1, keepdim=True)
    cov = torch.einsum("tsi, tsj -> tij", centered_pos, centered_pos)
    L, Q = torch.linalg.eigh(cov)
    if not allow_flip:
        d = torch.sign(torch.linalg.det(Q))
        Q[..., 0] = d.reshape(-1, 1) * Q[..., 0]
    
    rotated_pos = torch.einsum("tij, taj -> tai", Q, centered_pos)
    return rotated_pos

vmap_align_point_cloud_no_flips = torch.vmap(lambda x: align_point_cloud(x, allow_flip=False), in_dims=(0,))

def rotate_about_axis(vectors, axis, angle):
    '''Rotates vectors about an axis by an angle.
    Args:
        vectors (torch.Tensor of shape (n_vectors, 3)): The vectors to rotate.
        axis (torch.Tensor of shape (3,)): The axis of rotation: must be normalized!!
        angle (float): The angle of rotation, using the right-hand rule.
        
    Returns:
        vectors_rotated torch.Tensor of shape (n_vectors, 3): The rotated vectors.
    '''
    vectors_rotated = vectors * torch.cos(angle) + torch.cross(axis.unsqueeze(0), vectors, dim=1) * torch.sin(angle) + axis.unsqueeze(0) * (vectors @ axis).unsqueeze(1) * (1.0 - torch.cos(angle))
    return vectors_rotated

vmap_rotate_about_axis = torch.vmap(rotate_about_axis, in_dims=(0, 0, 0))

def rotate_about_axes(vectors, axis, angle):
    '''Rotates vectors about an axis by an angle.
    Args:
        vectors (torch.Tensor of shape (n_vectors, 3)): The vectors to rotate.
        axis (torch.Tensor of shape (n_vectors, 3)): The axis of rotation: must be normalized!!
        angle (torch.Tensor of shape (n_vectors,)): The angles of rotation, using the right-hand rule.
        
    Returns:
        vectors_rotated torch.Tensor of shape (n_vectors, 3): The rotated vectors.
    '''
    vectors_rotated = vectors * torch.cos(angle.unsqueeze(1)) + torch.cross(axis, vectors, dim=1) * torch.sin(angle.unsqueeze(1)) + axis * torch.sum(vectors * axis, dim=1, keepdim=True) * (1.0 - torch.cos(angle.unsqueeze(1)))
    return vectors_rotated

def to_torch_sparse(csc_sp):
    '''Takes a scipy sparse matrix in CSC format and converts it to a torch sparse tensor'''
    csc_torch = torch.sparse_csc_tensor(torch.tensor(csc_sp.indptr), torch.tensor(csc_sp.indices), torch.tensor(csc_sp.data), dtype=torch.float64)
    return csc_torch

def to_scipy_sparse(csc_torch):
    '''Takes a torch sparse tensor and converts it to a scipy sparse matrix in CSC format: antagonist to to_torch_sparse'''
    csc_sp = csc_matrix((csc_torch.values().numpy(), csc_torch.indices().numpy(), csc_torch.indptr().numpy()), shape=csc_torch.shape())
    return csc_sp

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def smooth_hat_function(x):
    return 2.0 * torch.maximum(1.0 - torch.abs(x), torch.tensor(0.0)) ** 3 - 8.0 * torch.maximum(0.5 - torch.abs(x), torch.tensor(0.0)) ** 3
def smooth_hat_function_vary_midpoint(x, midpoint):
    alpha = - np.log(2.0) / np.log(midpoint)
    return smooth_hat_function(x ** alpha)

def sigmoid_min_max(logits, min_val, max_val):
    '''Applies a sigmoid to logits and scales the result to the range [min_val, max_val]'''
    return min_val + (max_val - min_val) * torch.sigmoid(logits)