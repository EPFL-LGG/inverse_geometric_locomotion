import igl
import numpy as np
import pygmsh

def create_icosahedron(radius):
    '''
    Returns:
        vertices: np.array of shape (12, 3)
        faces: np.array of shape (20, 3)
    '''
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ])
    vertices = radius * vertices / np.linalg.norm(vertices[0])
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ])
    return vertices, faces

def subdivide(vertices, faces, radius):
    '''
    Args:
        vertices: np.array of shape (N, 3)
        faces: np.array of shape (M, 3)
        radius: float
    
    Returns:
        new_vertices: np.array of shape (N + M, 3)
        new_faces: np.array of shape
    '''
    vertex_map = {}
    new_faces = []
    new_vertices = vertices.tolist()

    def midpoint(v1, v2):
        if (v1, v2) in vertex_map:
            return vertex_map[(v1, v2)]
        elif (v2, v1) in vertex_map:
            return vertex_map[(v2, v1)]
        else:
            new_vertex = (vertices[v1] + vertices[v2]) / 2
            new_vertex_length = np.linalg.norm(new_vertex)
            new_vertex = (new_vertex / new_vertex_length) * radius
            new_vertices.append(new_vertex)
            index = len(new_vertices) - 1
            vertex_map[(v1, v2)] = index
            return index

    for face in faces:
        v1, v2, v3 = face
        a = midpoint(v1, v2)
        b = midpoint(v2, v3)
        c = midpoint(v3, v1)
        new_faces.extend([
            [v1, a, c],
            [v2, b, a],
            [v3, c, b],
            [a, b, c],
        ])
    return np.array(new_vertices), np.array(new_faces)

def generate_sphere_mesh(radius, subdivisions):
    '''
    Args:
        radius: float
        subdivisions: int
        
    Returns:
        vertices: np.array of shape (N, 3)
        faces: np.array of shape (M, 3)
    '''
    vertices, faces = create_icosahedron(radius)
    for _ in range(subdivisions):
        vertices, faces = subdivide(vertices, faces, radius)
    return vertices, faces

def generate_disk_mesh(radius, mesh_size, num_sections):
    '''
    Args:
        radius: float, the radius of the disk
        mesh_size: float, the size of the mesh
        num_sections: int, the number of sections
        
    Returns:
        vertices: np.array of shape (N, 3)
        faces: np.array of shape (M, 3)
    '''
    with pygmsh.geo.Geometry() as geom:
        geom.add_circle(
            [0.0, 0.0, 0.0],
            radius,
            mesh_size=mesh_size,
            num_sections=num_sections,
            # If compound==False, the section borders have to be points of the
            # discretization. If using a compound circle, they don't; gmsh can
            # choose by itself where to point the circle points.
            compound=False,
        )
        mesh = geom.generate_mesh()
        
    vertices, faces, _, _ = igl.remove_unreferenced(np.array(mesh.points, dtype=np.float64), np.array(mesh.cells_dict["triangle"], dtype=np.int32))
    return vertices, faces
