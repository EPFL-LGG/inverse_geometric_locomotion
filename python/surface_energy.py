import igl
import numpy as np
import torch

###############################################
# Functions related to connectivity
###############################################

def get_internal_edges_and_flaps(faces):
    '''
    Args:
        faces: np.array of shape (M, 3)
        
    Returns:
        i_edges: np.array of shape (N, 2) the internal edges
        i_edge_flaps: np.array of shape (N, 2) the faces adjacent to the internal edges
        i_edge_flap_corners: np.array of shape (N, 2) the corners of the faces adjacent to the internal edges
        idx_i_edge: np.array of shape (N,) the indices of the internal edges in the edges array
        edges: np.array of shape (n_all_edges, 2) the edges of the mesh
        edge_map: np.array of shape (M*3) the mapping between the edges and the faces
    '''
    
    edges, edge_map, edge_flaps, edge_flap_corners = igl.edge_flaps(faces)

    idx_i_edge = np.argwhere(np.min(edge_flaps, axis=1) != -1).reshape(-1,)
    i_edges = edges[idx_i_edge]
    i_edge_flaps = edge_flaps[idx_i_edge]
    i_edge_flap_corners = edge_flap_corners[idx_i_edge]
    
    return i_edges, i_edge_flaps, i_edge_flap_corners, idx_i_edge, edges, edge_map

###############################################
# Functions related to geometry processing
###############################################

def compute_mesh_face_normals(vertices, faces):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        
    Returns:
        face_normals: torch.tensor of shape (M, 3)
    '''
    
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = torch.cross(v1, v2, dim=1)
    face_normals = torch.nn.functional.normalize(
        face_normals, eps=1.0e-8, dim=1
    )
    return face_normals

vmap_compute_mesh_face_normals = torch.vmap(compute_mesh_face_normals, in_dims=(0, None))

def compute_mesh_diahedral_angles(face_normals, triangle_adjacency):
    '''
    Args:
        face_normals: torch.tensor of shape (n_faces, 3)
        triangle_adjacency: torch.tensor of shape (n_face_pairs, 2)
        
    Returns:
        dihedral_angles: torch.tensor of shape (n_face_pairs, 3)
    '''
    c = torch.sum(face_normals[triangle_adjacency[:, 0]] * face_normals[triangle_adjacency[:, 1]], dim=-1)
    s = torch.linalg.norm(torch.cross(face_normals[triangle_adjacency[:, 0]], face_normals[triangle_adjacency[:, 1]] + 1.0e-8, dim=-1), dim=-1)
    return torch.arctan2(s, c)
    
vmap_compute_mesh_diahedral_angles = torch.vmap(compute_mesh_diahedral_angles, in_dims=(0, None))

def compute_mesh_vertex_normals(vertices, faces):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        
    Returns:
        vertex_normals: torch.tensor of shape (N, 3)
    '''
    
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = torch.cross(v1, v2, dim=1) # the norm is equal to twice the area of the face
    
    # Vertex normals will be weighted by the area of the faces
    vertex_normals = torch.zeros(vertices.shape, dtype=torch.float64)
    vertex_normals = vertex_normals.index_add(
        0, faces[:, 0], face_normals
    )
    vertex_normals = vertex_normals.index_add(
        0, faces[:, 1], face_normals
    )
    vertex_normals = vertex_normals.index_add(
        0, faces[:, 2], face_normals
    )

    vertex_normals = torch.nn.functional.normalize(
        vertex_normals, eps=1.0e-6, dim=1
    )

    return vertex_normals

vmap_compute_mesh_vertex_normals = torch.vmap(compute_mesh_vertex_normals, in_dims=(0, None))

def compute_voronoi_areas(vertices, faces):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        
    Returns:
        voronoi_areas: torch.tensor of shape (N, 1)
    '''
    
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = torch.cross(v1, v2, dim=1) # the norm is equal to twice the area of the face
    
    voronoi_areas = torch.zeros(size=(vertices.shape[0],), dtype=torch.float64)
    voronoi_areas = voronoi_areas.index_add(
        0, faces.reshape(-1,), torch.repeat_interleave(torch.linalg.norm(face_normals, dim=1), 3)
    )
    
    voronoi_areas = voronoi_areas / 3.0
    return voronoi_areas

vmap_compute_voronoi_areas = torch.vmap(compute_voronoi_areas, in_dims=(0, None))

def compute_mesh_volume(vertices, faces):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        
    Returns:
        volume: torch.tensor of shape (,)
    '''
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return torch.einsum('ij, ij', v0, torch.cross(v1, v2, dim=1)) / 6.0

vmap_compute_mesh_volume = torch.vmap(compute_mesh_volume, in_dims=(0, None))

def compute_mesh_face_areas(vertices, faces):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        
    Returns:
        face_areas: torch.tensor of shape (M,)
    '''
    v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return 0.5 * torch.linalg.norm(torch.cross(v1, v2, dim=1), dim=1)

vmap_compute_mesh_face_areas = torch.vmap(compute_mesh_face_areas, in_dims=(0, None))

def compute_mesh_surface_area(vertices, faces):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        
    Returns:
        surface_area: torch.tensor of shape (,)
    '''
    return torch.sum(compute_mesh_face_areas(vertices, faces))

vmap_compute_mesh_surface_area = torch.vmap(compute_mesh_surface_area, in_dims=(0, None))

def compute_weighted_mesh_surface_area(vertices, faces, weights):
    '''
    Args:
        vertices: torch.tensor of shape (N, 3)
        faces: torch.tensor of shape (M, 3)
        weights: torch.tensor of shape (M,)
        
    Returns:
        surface_area: torch.tensor of shape (,)
    '''
    return torch.sum(weights * compute_mesh_face_areas(vertices, faces))

vmap_compute_weighted_mesh_surface_area = torch.vmap(compute_weighted_mesh_surface_area, in_dims=(0, None, None))

def compute_jacobian(vertices, faces):
    '''
    Args:
        vertices: (n_v, 3) tensor representing the vertices of the mesh
        faces: (n_f, 3) tensor representing the faces of the mesh
        
    Returns:
        (n_f, 3, 2) tensor representing the Jacobian of the mesh
    '''
    return torch.stack([
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    ], dim=2)
    
def compute_first_fundamental_form(vertices, faces):
    '''
    Args:
        vertices: (n_v, 3) tensor representing the vertices of the mesh
        faces: (n_f, 3) tensor representing the faces of the mesh
        
    Returns:
        (n_f, 2, 2) tensor representing the first fundamental form of the mesh
    '''
    jac = compute_jacobian(vertices, faces)
    return torch.einsum('ijk,ijl->ikl', jac, jac)

class VertexEltsSum:
    
    def __init__(self, nv, elts, row_major=True):
        '''
        Args:
            nv: number of vertices
            elts: torch tensor of shape (#e, 3 or 4), containing the indices of the vertices of each element of the mesh
        '''
        self.row_major = row_major
        if row_major:
            self.i  = elts.flatten()
            self.j  = torch.repeat_interleave(torch.arange(elts.shape[0]), elts.shape[1], dim=0).to(elts.device)
        else:
            self.i  = elts.T.flatten()
            self.j  = torch.tile(torch.arange(elts.shape[0]), (elts.shape[1],)).to(elts.device)
    
        self.indices = torch.stack([self.i, self.j], dim=0)
        self.nv = nv
        self.ne = elts.shape[0]
        self.vert_per_elt = elts.shape[1]
        
    def vertex_elt_sum(self, data):
        '''
        Distributes data specified at each element to the neighboring vertices.
        All neighboring vertices will receive the value indicated at the corresponding surface/tet position in data.

        Args:
            data: torch tensor of shape ((3 or 4) * #e,), flattened in a column-major fashion

        Returns:
            data_sum: torch array of shape (#v,), containing the summed data
        '''
        v_sum = torch.sparse_coo_tensor(self.indices, data, (self.nv, self.ne))
        return torch.sparse.mm(v_sum, torch.ones(size=(self.ne, 1))).flatten()

###############################################
# Functions related to energies
###############################################

def compute_membrane_energy_per_face(vertices, faces, delta, lambda_, mu, ref_inv_ff, ref_areas):
    '''
    Args:
        vertices: (n_v, 3) tensor representing the vertices of the mesh
        faces: (n_f, 3) tensor representing the faces of the mesh
        delta: float or (n_f,) representing the width of the shell
        lambda_: first lame parameter
        mu: second lame parameter
        ref_inv_ff: (n_f, 3, 3) tensor representing the inverse of the reference first fundamental form
        ref_areas: (n_f,) tensor representing the reference areas of the faces
        
    Returns:
        (n_f,) tensor representing the membrane energy of the mesh per face
    '''
    
    ff = compute_first_fundamental_form(vertices, faces)
    strain = torch.einsum('ijk,ikl->ijl', ref_inv_ff, ff)
    det_strain = torch.det(strain)
    energy_density = mu / 2.0 * torch.einsum('ijj->i', strain) + lambda_ / 4.0 * det_strain - (mu / 2.0 + lambda_ / 4.0) * torch.log(det_strain + 1.0e-8)  - mu - lambda_ / 4.0
    
    return delta * energy_density * ref_areas

def compute_membrane_energy(vertices, faces, delta, lambda_, mu, ref_inv_ff, ref_areas):
    '''
    Args:
        Same as above
        
    Returns:
        float representing the membrane energy of the mesh
    '''
    return torch.sum(compute_membrane_energy_per_face(vertices, faces, delta, lambda_, mu, ref_inv_ff, ref_areas))

vmap_compute_membrane_energy_per_face = torch.vmap(compute_membrane_energy_per_face, in_dims=(0, None, None, None, None, None, None))
vmap_compute_membrane_energy = torch.vmap(compute_membrane_energy, in_dims=(0, None, None, None, None, None, None))

def compute_bending_energy_per_int_edge(vertices, faces, face_adjacency, delta, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''
    Args:
        vertices: (n_v, 3) tensor representing the vertices of the mesh
        faces: (n_f, 3) tensor representing the faces of the mesh
        face_adjacency: (n_face_pairs, 2) array of the face adjacency
        delta: float or (n_face_pairs,) representing the width of the shell
        ref_diahedral_angles: (n_face_pairs,) tensor representing the reference dihedral angles
        ref_edge_lengths_sq: (n_face_pairs,) tensor representing the reference edge squared lengths between adjacent faces
        ref_edge_areas: (n_face_pairs,) tensor representing the reference edge areas between adjacent faces
        
    Returns:
        (n_face_pairs,) tensor representing the bending energy of the mesh per face
    '''
    
    face_normals = compute_mesh_face_normals(vertices, faces)
    diahedral_angles = compute_mesh_diahedral_angles(face_normals, face_adjacency)
    energy_density = (2.0 * torch.tan(diahedral_angles / 2.0) - 2.0 * torch.tan(ref_diahedral_angles / 2.0)) ** 2
    
    return delta ** 3 * energy_density * ref_edge_lengths_sq / ref_edge_areas

def compute_bending_energy(vertices, faces, face_adjacency, delta, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''
    Args:
        Same as above
        
    Returns:
        float representing the bending energy of the mesh
    '''
    return torch.sum(compute_bending_energy_per_int_edge(vertices, faces, face_adjacency, delta, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas))

vmap_compute_bending_energy_per_int_edge = torch.vmap(compute_bending_energy_per_int_edge, in_dims=(0, None, None, None, None, None, None))
vmap_compute_bending_energy = torch.vmap(compute_bending_energy, in_dims=(0, None, None, None, None, None, None))

def compute_bending_energy_small_angles_per_int_edge(vertices, faces, face_adjacency, delta, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''
    Args:
        vertices: (n_v, 3) tensor representing the vertices of the mesh
        faces: (n_f, 3) tensor representing the faces of the mesh
        face_adjacency: (n_face_pairs, 2) array of the face adjacency
        delta: float or (n_face_pairs,) representing the width of the shell at the edges 
        ref_diahedral_angles: (n_face_pairs,) tensor representing the reference dihedral angles
        ref_edge_lengths_sq: (n_face_pairs,) tensor representing the reference edge squared lengths between adjacent faces
        ref_edge_areas: (n_face_pairs,) tensor representing the reference edge areas between adjacent faces
        
    Returns:
        (n_face_pairs,) tensor representing the bending energy of the mesh per face
    '''
    
    face_normals = compute_mesh_face_normals(vertices, faces)
    diahedral_angles = compute_mesh_diahedral_angles(face_normals, face_adjacency)
    energy_density = (diahedral_angles - ref_diahedral_angles) ** 2
    return delta ** 3 * energy_density * ref_edge_lengths_sq / ref_edge_areas

def compute_bending_energy_small_angles(vertices, faces, face_adjacency, delta, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas):
    '''
    Args:
        Same as above
        
    Returns:
        float representing the bending energy of the mesh
    '''
    return torch.sum(compute_bending_energy_small_angles_per_int_edge(vertices, faces, face_adjacency, delta, ref_diahedral_angles, ref_edge_lengths_sq, ref_edge_areas))

vmap_compute_bending_energy_small_angles_per_int_edge = torch.vmap(compute_bending_energy_small_angles_per_int_edge, in_dims=(0, None, None, None, None, None, None))
vmap_compute_bending_energy_small_angles = torch.vmap(compute_bending_energy_small_angles, in_dims=(0, None, None, None, None, None, None))