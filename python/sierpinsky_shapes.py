import os
import sys as _sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CUBICSPLINES = os.path.join(os.path.dirname(SCRIPT_PATH), 'ext/torchcubicspline')
_sys.path.append(PATH_TO_CUBICSPLINES)

import numpy as np
import torch
from torchcubicspline import (
    natural_cubic_spline_coeffs, NaturalCubicSpline
)

import torch
from utils import rotate_about_axes

def sierpinski_triangle_dense(iterations):
    '''
    Args:
        iteration (int): the number of subdivisions applied to the equilateral triangle
        
    Returns:
        vertices: (n_vertices, 2) numpy array representing the vertices of the sierpinsky triangle in the 2D plane
        faces: (n_faces, 3) numpy array representing the faces of the sierpinsky triangle
        face_ids: (n_faces,) numpy array representing the id of the big triangle each face belongs to
        vertex_face_ids: (n_vertices,) numpy array representing the id of the big triangle each vertex belongs to (3 for the center vertices)
        hinge_vertex_ids: (3, 2) numpy array representing the vertices forming each hinge in the triangle
    '''
    assert iterations > 0, "There should be at least one hinge"
    # Initial vertices of the equilateral triangle
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]]) - 1.0 / 3.0 * np.ones(shape=(1, 2))
    faces = np.array([[0, 1, 2]])
    # What big triangle does a face belong to?
    face_ids = np.array([0], dtype=np.int32)
    
    for id_iter in range(iterations):
        new_faces = []
        new_face_ids = []
        for face, face_id in zip(faces, face_ids):
            v0, v1, v2 = vertices[face]
            # Midpoints of each side of the triangle
            v01 = (v0 + v1) / 2.0
            v12 = (v1 + v2) / 2.0
            v20 = (v2 + v0) / 2.0
            
            # Add new vertices
            vertices = np.vstack([vertices, v01, v12, v20])
            i = len(vertices) - 3
            new_faces.extend([
                [face[0], i, i+2],
                [i, face[1], i+1],
                [i+2, i+1, face[2]],
                [i, i+1, i+2]
            ])
            if id_iter == 0:
                new_face_ids = [0, 1, 2, 3]
            else:
                new_face_ids.extend([face_id, face_id, face_id, face_id])
            
        faces = np.array(new_faces)
        face_ids = np.array(new_face_ids, dtype=np.int32)
        
    vertex_face_ids = [None for _ in range(vertices.shape[0])]
    for face, face_id in zip(faces, face_ids):
        for vid in face:
            if vertex_face_ids[vid] is None:
                vertex_face_ids[vid] = face_id
    vertex_face_ids = np.array(vertex_face_ids, dtype=np.int32)
                
    # 3/0, 3/1, 3/2 hinges
    hinge_vertex_ids = [[3, 5], [4, 3], [5, 4]]
    
    return vertices, faces, face_ids, vertex_face_ids, hinge_vertex_ids

def deform_sierpinski_triangle(dihedral_angles, vertices, vertex_face_ids, hinge_vertex_ids):
    '''
    Args:
        dihedral_angles (torch.tensor of shape (3,)): the dihedral angles associated to each of the outer triangles
        vertices (torch.tensor of shape (n_vertices, 3)): the vertices of the sierpinsky triangle
        vertex_face_ids (torch.tensor of shape (n_vertices,)): the dihedral angle to use (integers between 0 and 2 included)
        hinge_vertex_ids (torch.tensor of shape (3, 2)): the vertices forming each hinge in the triangle
        
    Returns:
        vertices_transformed (torch.tensor of shape (n_vertices, 3)): the vertices once the rotations about the hinges have been applied
    '''

    hinge_vertices = vertices[hinge_vertex_ids]
    hinge_directions = hinge_vertices[:, 1] - hinge_vertices[:, 0]
    hinge_directions = hinge_directions / torch.linalg.norm(hinge_directions, dim=-1, keepdim=True)
    
    mask_deformed_vertices = torch.le(vertex_face_ids, 2)
    hinge_vertices_per_vertex_id = hinge_vertices[vertex_face_ids[mask_deformed_vertices]]
    
    deformed_vertices = vertices
    deformed_vertices[mask_deformed_vertices] = rotate_about_axes(vertices[mask_deformed_vertices] - hinge_vertices_per_vertex_id[:, 0], hinge_directions[vertex_face_ids[mask_deformed_vertices]], dihedral_angles[vertex_face_ids[mask_deformed_vertices]]) + hinge_vertices_per_vertex_id[:, 0]
    
    return deformed_vertices

vmap_deform_sierpinski_triangle = torch.vmap(deform_sierpinski_triangle, in_dims=(0, 0, None, None))

def sierpinski_deformation_from_cubic_spline(
    control_points, vertices_ref, vertex_face_ids, hinge_vertex_ids, n_ts, n_cp, close_gait=False,
):
    '''Generate deformations of the sierpinski triangle, parameterized by its dihedral angles
    
    Args:
        control_points (torch.tensor of shape (n_cps, 3)): the diahedral angles that serve parameterizing the cubic spline
        n_ts: int representing the number of time steps
        n_cp: int representing the number of control points
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''
    
    t = torch.linspace(0.0, 1.0, n_ts)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_gait)

    if close_gait:
        control_points = torch.cat([control_points, control_points[0].reshape(1, -1)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points)
    spline = NaturalCubicSpline(spline_coeffs)
    dihedral_angles = spline.evaluate(t)
    
    tiled_vertices_ref = torch.tile(torch.tensor(vertices_ref).reshape(1, -1, 3), (n_ts, 1, 1))
    
    return vmap_deform_sierpinski_triangle(dihedral_angles, tiled_vertices_ref, torch.tensor(vertex_face_ids), torch.tensor(hinge_vertex_ids))