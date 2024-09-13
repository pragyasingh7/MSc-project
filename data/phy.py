import networkx as nx
import torch
import random


def convert_edge_feat_array_to_matrix(edge_feats, edge_idx, num_nodes):
    edge_feat_matrix = torch.zeros(num_nodes, num_nodes).to(edge_feats.device)
    for i, (u, v) in enumerate(edge_idx):
        edge_feat_matrix[u, v] = edge_feats[i]

    edge_feat_matrix = edge_feat_matrix + edge_feat_matrix.T  # Make the matrix symmetric
    
    return edge_feat_matrix

# Define the function for edge features
def get_target_edge_value(node_feat_i, node_feat_j, eq_type='e1'):
    # Each node_feat: [x coordinate, y coordinate, mass]
    # Use gravitational force as edge feature
    m1 = node_feat_i[-1]
    m2 = node_feat_j[-1]

    x1, y1 = node_feat_i[:-1]
    x2, y2 = node_feat_j[:-1]

    # Inverse square law for gravitational force
    if eq_type == 'e1':
        G = 0.1
        d = node_feat_i[:-1] - node_feat_j[:-1]
        r_2 = sum(d*d)
        return G * m1 * m2 / r_2
    
    # Asymmetric rational function
    if eq_type == 'e2':
        A = 10
        B = -7

        num = A*m1  + B*m2
        dem = x1**2 + y2**2

        return num / dem
    
    # Linear symmetric function 1
    if eq_type == 'e3':
        d = node_feat_i - node_feat_j
        return sum(d*d)
    
    # Linear symmetric function 2
    if eq_type == 'e4':
        return x1*y1*m1  + x2*y2*m2 + x1*y2 + x2*y1

    # Linear asymmetric function
    if eq_type == 'e5':
        return x1*x1 + y1*y1 + m2*m2

    # Linear asymmetric function 2
    if eq_type == 'e6':
        return x2*x2 + y2*y2 + m1*m1


def convert_edge_feat_array_to_matrix(edge_feats, edge_idx, num_nodes):
    edge_feat_matrix = torch.zeros(num_nodes, num_nodes).to(edge_feats.device)
    for i, (u, v) in enumerate(edge_idx):
        edge_feat_matrix[u, v] = edge_feats[i]

    edge_feat_matrix = edge_feat_matrix + edge_feat_matrix.T  # Make the matrix symmetric
    
    return edge_feat_matrix


# Create grid graph dataset
def create_grid_graphs(n_samples=100, grid_size=4, uniform_mass=False, eq_type='e1', device='cpu'):
    G = nx.grid_2d_graph(grid_size, grid_size)  # 4x4 grid
    num_nodes = G.number_of_nodes()
    A = nx.adjacency_matrix(G)

    node_to_idx_map = {n: i for i, n in enumerate(G.nodes())}
    edge_idx = torch.tensor([[node_to_idx_map[edge[0]], node_to_idx_map[edge[1]]] for edge in G.edges()]).to(device)

    graphs = []

    for _ in range(n_samples):
        node_pos = torch.tensor([n for n in G.nodes()], dtype=torch.float).to(device)  # Use fixed node positions
        if uniform_mass:
            node_mass = torch.ones((node_pos.shape[0], 1)).to(device)
        else:
            # Assign interger masses randomly
            node_mass = torch.rand((node_pos.shape[0], 1)).to(device)  # Randomly assign node masses
        node_feats = torch.cat([node_pos, node_mass], dim=1)

        edge_feats = torch.tensor([get_target_edge_value(node_feats[i], node_feats[j], eq_type) for i, j in edge_idx], 
                                    dtype=torch.float,
                                    device=device)

        graphs.append((node_feats, edge_idx.T, edge_feats, A, num_nodes))

    return graphs


# Create random geometric graph dataset

def create_random_geometric_graphs(n_samples=100, num_nodes=16, threshold=0.3, uniform_mass=False, 
                                   data_gen_seed=42, eq_type='e1', device='cpu'):
    graphs = []

    # Set seed for reproducibility
    random.seed(data_gen_seed)

    for _ in range(n_samples):
        G = nx.random_geometric_graph(num_nodes, threshold) 
        A = nx.adjacency_matrix(G)

        edge_idx = torch.tensor([e for e in G.edges()]).to(device)
        
        nodes = G.nodes(data=True)
        node_pos = torch.tensor([v['pos'] for k, v in nodes], dtype=torch.float).to(device)   # Random node positions

        if uniform_mass:
            node_mass = torch.ones((node_pos.shape[0], 1), device=device)
        else:
            # Assign interger masses randomly
            node_mass = torch.rand((node_pos.shape[0], 1), device=device)  # Randomly assign node masses

        node_feats = torch.cat([node_pos, node_mass], dim=1)

        edge_feats = torch.tensor([get_target_edge_value(node_feats[i], node_feats[j], eq_type) for i, j in edge_idx], 
                                dtype=torch.float, device=device)

        graphs.append((node_feats, edge_idx.T, edge_feats, A, num_nodes))

    return graphs