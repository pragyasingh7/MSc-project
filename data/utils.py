import gc
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from node2vec import Node2Vec


def create_pyg_simulated_nodes(x, adj, device='cpu'):
    pos_edge_index = adj.nonzero(as_tuple=False).t().contiguous().to(device)
    pyg_data = Data(x=x, pos_edge_index=pos_edge_index)
    return pyg_data


def create_pyg_graph(x, n_nodes, node_feature_init='adj', node_feat_dim=1):
    """
    Create a PyTorch Geometric graph data object from given adjacency matrix.
    """
    # Initialise edge features
    if isinstance(x, torch.Tensor):
        edge_attr = x.view(-1, 1)
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)

    # Initialise node features
    # 1. From adjacency matrix
    if node_feature_init == 'adj':
        if isinstance(x, torch.Tensor):
            # node_feat = x.clone().detach()
            node_feat = x
        else:
            node_feat = torch.tensor(x, dtype=torch.float)

    # 2. Random initialisation
    elif node_feature_init == 'random':
        node_feat = torch.randn(n_nodes, node_feat_dim, device=edge_attr.device)

    # 3. Ones initialisation
    elif node_feature_init == 'ones':
        node_feat = torch.ones(n_nodes, node_feat_dim, device=edge_attr.device)

    # 4. Identity initialisation
    elif node_feature_init == 'identity':
        node_feat = torch.eye(n_nodes, device=edge_attr.device)

    else:
        raise ValueError(f"Unsupported node feature initialization: {node_feature_init}")


    rows, cols = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0).to(edge_attr.device)

    pyg_graph = Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
    
    return pyg_graph


class MatrixVectorizer:
    """
    A class for transforming between matrices and vector representations.
    
    This class provides methods to convert a symmetric matrix into a vector (vectorize)
    and to reconstruct the matrix from its vector form (anti_vectorize), focusing on 
    vertical (column-based) traversal and handling of elements.
    """

    def __init__(self):
        """
        Initializes the MatrixVectorizer instance.
        
        The constructor currently does not perform any actions but is included for 
        potential future extensions where initialization parameters might be required.
        """
        pass

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        """
        Converts a matrix into a vector by vertically extracting elements.
        
        This method traverses the matrix column by column, collecting elements from the
        upper triangle, and optionally includes the diagonal elements immediately below
        the main diagonal based on the include_diagonal flag.
        
        Parameters:
        - matrix (numpy.ndarray): The matrix to be vectorized.
        - include_diagonal (bool, optional): Flag to include diagonal elements in the vectorization.
          Defaults to False.
        
        Returns:
        - numpy.ndarray: The vectorized form of the matrix.
        """
        # Determine the size of the matrix based on its first dimension
        matrix_size = matrix.shape[0]

        # Initialize an empty list to accumulate vector elements
        vector_elements = []

        # Iterate over columns and then rows to collect the relevant elements
        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:  
                    if row < col:
                        # Collect upper triangle elements
                        vector_elements.append(matrix[row, col])
                    elif include_diagonal and row == col + 1:
                        # Optionally include the diagonal elements immediately below the diagonal
                        vector_elements.append(matrix[row, col])

        return np.array(vector_elements)

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        """
        Reconstructs a matrix from its vector form, filling it vertically.
        
        The method fills the matrix by reflecting vector elements into the upper triangle
        and optionally including the diagonal elements based on the include_diagonal flag.
        
        Parameters:
        - vector (numpy.ndarray): The vector to be transformed into a matrix.
        - matrix_size (int): The size of the square matrix to be reconstructed.
        - include_diagonal (bool, optional): Flag to include diagonal elements in the reconstruction.
          Defaults to False.
        
        Returns:
        - numpy.ndarray: The reconstructed square matrix.
        """
        # Initialize a square matrix of zeros with the specified size
        matrix = np.zeros((matrix_size, matrix_size))

        # Index to keep track of the current position in the vector
        vector_idx = 0

        # Fill the matrix by iterating over columns and then rows
        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:  
                    if row < col:
                        # Reflect vector elements into the upper triangle and its mirror in the lower triangle
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1
                    elif include_diagonal and row == col + 1:
                        # Optionally fill the diagonal elements after completing each column
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1

        return matrix



def create_dual_graph(adjacency_matrix):
    """Returns edge_index and node_feature_matrix for the undirected dual graph"""
    # Number of nodes in the original graph G1
    n = adjacency_matrix.shape[0]

    # Find all potential edges in the upper triangular part
    row, col = torch.triu_indices(n, n, offset=1)
    all_edges = torch.stack([row, col], dim=1).to(adjacency_matrix.device)  # gpu
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1)   # gpu

    # Filter actual edges
    actual_edges = all_edges[actual_edges_mask] # gpu

    # Number of edges in G1
    num_actual_edges = actual_edges.shape[0]
    max_possible_edges = row.size(0)

    # Create a tensor indicating shared nodes between edges
    edge_to_nodes = torch.zeros((max_possible_edges, n), dtype=torch.float, device=adjacency_matrix.device)
    edge_to_nodes[actual_edges_mask, actual_edges[:, 0]] = 1.0
    edge_to_nodes[actual_edges_mask, actual_edges[:, 1]] = 1.0  # gpu

    # Compute the connectivity between edges
    shared_nodes_matrix = edge_to_nodes @ edge_to_nodes.t() # gpu
    shared_nodes_matrix.fill_diagonal_(0)  # Remove self-loops

    # Extract edge indices from the connectivity matrix
    edge_index = shared_nodes_matrix.nonzero(as_tuple=False).t().contiguous()   # gpu

    # Create node feature matrix for the dual graph
    node_feat_matrix = torch.zeros((max_possible_edges, 1), dtype=torch.float, device=adjacency_matrix.device)  # gpu
    node_feat_matrix[actual_edges_mask] = adjacency_matrix[actual_edges[:, 0], actual_edges[:, 1]].view(-1, 1).float()  # gpu

    torch.cuda.empty_cache()
    gc.collect()

    return edge_index, node_feat_matrix


def create_dual_graph_feature_matrix(adjacency_matrix):
    """Returns node_feature_matrix for the dual graph."""
    # Number of nodes in the original graph G1
    n = adjacency_matrix.shape[0]

    # Find all potential edges in the upper triangular part
    row, col = torch.triu_indices(n, n, offset=1)
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1).cpu()

    # Create node feature matrix for the dual graph
    node_feat_matrix = torch.zeros((row.size(0), 1), dtype=torch.float, device=adjacency_matrix.device)
    node_feat_matrix[actual_edges_mask] = adjacency_matrix[row[actual_edges_mask], col[actual_edges_mask]].view(-1, 1).float()

    return node_feat_matrix


def revert_dual(node_feat, n_nodes):
    """Reverts the dual node feature matrix to the original adjacency matrix."""
    # node_feat: (n_t*(n_t-1)/2, 1)
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float, device=node_feat.device)
    row, col = torch.triu_indices(n_nodes, n_nodes, offset=1)
    adj[row, col] = node_feat.view(-1)
    adj[col, row] = node_feat.view(-1)  # (n_t, n_t)

    torch.cuda.empty_cache()
    gc.collect()
    
    return adj


def graph_to_pyg_nodes(G, feat_type='one-hot', x_dim=16, 
                       n2v_walk_length=0.5, n2v_num_walks=100, 
                       n2v_p=1, n2v_q=1, device='cpu'):
    edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)

    if feat_type == 'one-hot':
        x = torch.tensor(np.eye(G.number_of_nodes())).float().to(device)
    elif feat_type == 'node2vec':
        n2v_walk_length = int(n2v_walk_length * G.number_of_nodes())
        node2vec = Node2Vec(G, dimensions=x_dim, 
                            walk_length=n2v_walk_length, num_walks=n2v_num_walks, 
                            workers=4, p=n2v_p, q=n2v_q)  # p, q are hyperparameters
        embeddings = node2vec.fit(window=10, min_count=1, batch_words=4).wv
        x = torch.tensor(embeddings.vectors).to(device)
    elif feat_type == 'adj':
        x = torch.tensor(nx.to_numpy_array(G)).float().to(device)
    elif feat_type == 'random':
        x = torch.randn(G.number_of_nodes(), x_dim).float().to(device)
    else:
        raise ValueError(f"Invalid feature type: {feat_type}")
    
    return Data(x=x, edge_index=edge_index)


def graph_to_node_feat(G, feat_type='one-hot', x_dim=16, 
                       n2v_walk_length=0.5, n2v_num_walks=100, 
                       n2v_p=1, n2v_q=1, device='cpu'):
    if feat_type == 'one-hot':
        x = torch.tensor(np.eye(G.number_of_nodes())).float().to(device)
    elif feat_type == 'node2vec':
        n2v_walk_length = int(n2v_walk_length * G.number_of_nodes())
        node2vec = Node2Vec(G, dimensions=x_dim, 
                            walk_length=n2v_walk_length, num_walks=n2v_num_walks, 
                            workers=4, p=n2v_p, q=n2v_q)  # p, q are hyperparameters
        embeddings = node2vec.fit(window=10, min_count=1, batch_words=4).wv
        x = torch.tensor(embeddings.vectors).to(device)
    elif feat_type == 'adj':
        x = torch.tensor(nx.to_numpy_array(G)).float().to(device)
    elif feat_type == 'random':
        x = torch.randn(G.number_of_nodes(), x_dim).float().to(device)
    else:
        raise ValueError(f"Invalid feature type: {feat_type}")

    return x