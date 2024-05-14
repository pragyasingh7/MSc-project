import torch
from torch_geometric.data import Data
import hydra
import networkx as nx
import numpy as np


def calc_n_nodes_f(n_source_nodes, n_target_nodes):
    """
    Calculate the number of edges in a fully connected graph.
    """
    n_source_nodes_f = int(n_source_nodes * (n_source_nodes - 1) / 2)
    n_target_nodes_f = int(n_target_nodes * (n_target_nodes - 1) / 2)

    return n_source_nodes_f, n_target_nodes_f

def create_pyg_graph(x, n_nodes):
    """
    Create a PyTorch Geometric graph data object from given adjacency matrix.
    """
    if isinstance(x, torch.Tensor):
        edge_attr = x.clone().detach().view(-1, 1)
        node_feat = x.clone().detach()
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)
        node_feat = torch.tensor(x, dtype=torch.float)

    rows, cols = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    pyg_graph = Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
    
    return pyg_graph

def create_networkx_graph(matrix):
    """
    Create a networkx graph from an adjacency matrix
    """
    # Transfer to CPU to work with networkx
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    # Zero out self edges
    np.fill_diagonal(matrix, 0)
    G = nx.from_numpy_array(np.absolute(matrix))

    # Make the graph undirected
    G = G.to_undirected()

    return G

def compute_topological_measures(matrix):
    """
    Compute topological measures from an adjacency matrix.
    """
    # Create networkx graph
    G = create_networkx_graph(matrix)

    # Compute closeness centrality
    # cc = nx.closeness_centrality(G, distance="weight")
    # cc = torch.tensor([cc[n] for n in G])

    # Compute betweenness centrality
    # bc = nx.betweenness_centrality(G, weight='weight')
    # bc = torch.tensor([bc[n] for n in G])

    # Compute eigenvector centrality
    ec = nx.eigenvector_centrality_numpy(G)
    ec = torch.tensor([ec[n] for n in G])

    # return cc, bc, ec
    return ec

def get_LR_from_HR(hr_matrix_list, scale=4, pooling_type='mean'):
    """
    Get LR matrix from HR matrix.
    """
    hr_matrix_list = [torch.tensor(hr_matrix) for hr_matrix in hr_matrix_list]
    hr_matrix = torch.stack(hr_matrix_list, dim=0)

    if pooling_type == 'mean':
        pool_operator = torch.nn.AvgPool2d(kernel_size=scale, stride=scale)
    elif pooling_type == 'max':
        pool_operator = torch.nn.MaxPool2d(kernel_size=scale, stride=scale)
    else:
        raise ValueError(f"Pooling type {pooling_type} not supported.")
    
    lr_matrix = pool_operator(hr_matrix)

    return lr_matrix.numpy()

