import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, depth_first_order
import torch

def compute_ordered_edges(edges, n_vertices, i_start=0):
    '''
    Compute the edges of the spanning tree in the order of a depth-first search.
    
    Args:
        edges: (n_edges, 2) array of the edges of the graph
        n_vertices: int, number of vertices in the graph
        i_start: int, index of the vertex to start the depth-first search
        
    Returns:
        edges_ordered: (n_edges, 2) array of the edges of the spanning tree
    '''
    id_i = np.concatenate([edges[:, 0], edges[:, 1]])
    id_j = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = csr_matrix((np.ones(shape=(2*edges.shape[0],)), (id_i, id_j)), shape=(n_vertices, n_vertices))
    
    tree = minimum_spanning_tree(graph)
    order, pred = depth_first_order(tree, i_start, directed=False, return_predecessors=True)
    edges_ordered = np.stack([pred[1:], order[1:]], axis=1)

    return edges_ordered

def compute_vertex_positions(pos0, edge_lengths, edge_angles, edges_ordered, n_vertices):
    '''
    Compute the positions of the vertices in the graph.
    
    Args:
        pos0: (?, 3) tensor of the position of edges_ordered[0, 0], to start tracing the graph
        edge_lengths: (?, n_edges,) tensor of the lengths of the edges
        edge_angles: (?, n_edges, 2) tensor of the angles of the edges
        edges_ordered: (n_edges, 2) tensor of the edges of the spanning tree
        
    Returns:
        pos: (?, n_vertices, 2) tensor of the positions of the vertices
    '''
    batch_dims = edge_lengths.shape[:-1]
    edges_dirs_rec = torch.zeros(size=batch_dims + (edges_ordered.shape[0], 3))
    edges_dirs_rec[..., 0] = torch.sin(edge_angles[..., 0]) * torch.cos(edge_angles[..., 1])
    edges_dirs_rec[..., 1] = torch.sin(edge_angles[..., 0]) * torch.sin(edge_angles[..., 1])
    edges_dirs_rec[..., 2] = torch.cos(edge_angles[..., 0])
    edges_segments_rec = edge_lengths.unsqueeze(-1) * edges_dirs_rec

    pos = torch.zeros(size=batch_dims + (n_vertices, 3))
    pos[..., edges_ordered[0, 0], :] = pos0
    for id_ed, ed in enumerate(edges_ordered):
        pos[..., ed[1], :] = pos[..., ed[0], :] + edges_segments_rec[..., id_ed, :]
    
    return pos
