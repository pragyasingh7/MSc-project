# Models to test whether node representation learning is sufficient to predict edge features. 
# In line with the literature, we use dot product of the learned embeddings as edge features.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class NodeModel(nn.Module):
    def __init__(self, gin_in_size=3, gin_hidden_size=16, gin_out_size=16, ):
        super().__init__()
        self.gin_nn = nn.Sequential(
            nn.Linear(gin_in_size, gin_hidden_size),
            nn.ReLU(),
            nn.Linear(gin_hidden_size, gin_out_size),
            nn.ReLU(),
        )
        self.gin = GINConv(self.gin_nn, train_eps=True)

    def forward(self, node_feats, edge_idx):
        # print("Node feats", node_feats.device)
        # print("Edge idx", edge_idx.device)

        h = self.gin(node_feats, edge_idx)

        # Compute edge features using dot product
        # edge_feats = torch.matmul(h[edge_idx[0]], h[edge_idx[1]].T)
        edge_feats = h @ h.T

        return edge_feats
    

class NodeLargeModel(nn.Module):
    def __init__(self, gin_in_size=3, gin_hidden_size=16, gin_out_size=16, ):
        super().__init__()
        self.gin_nn = nn.Sequential(
            nn.Linear(gin_in_size, gin_hidden_size),
            nn.ReLU(),
            nn.Linear(gin_hidden_size, gin_out_size),
            nn.ReLU(),
            nn.Linear(gin_out_size, 1),
            nn.ReLU()
        )
        self.gin = GINConv(self.gin_nn, train_eps=True)

    def forward(self, node_feats, edge_idx):
        h = self.gin(node_feats, edge_idx)

        # Compute edge features using dot product
        # edge_feats = torch.matmul(h[edge_idx[0]], h[edge_idx[1]].T)
        edge_feats = h @ h.T

        return edge_feats
    

class EdgeModel(nn.Module):
    def __init__(self, in_size=3, hidden_size1=16, hidden_size2=16 ):
        super().__init__()
        self.edge_nn = nn.Sequential(
            nn.Linear(2*in_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
            nn.ReLU()
        )

    def forward(self, node_feats, edge_idx):
        # Create edge features by concatenating node embeddings
        edge_feats = torch.cat([node_feats[edge_idx[0]], node_feats[edge_idx[1]]], dim=1)
        edge_feats = self.edge_nn(edge_feats).squeeze(1)
        return edge_feats


class DualEdgeModel(nn.Module):
    def __init__(self, in_size=3, hidden_size1=16, hidden_size2=16 ):
        super().__init__()
        self.dual_gin_nn = nn.Sequential(
            nn.Linear(2*in_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
            nn.ReLU()
        )

        self.gin = GINConv(self.dual_gin_nn, train_eps=True)

    def dual_edge_idx(self, edge_idx, num_nodes):
        n_edges = edge_idx.shape[1]
        edges_to_nodes = torch.zeros(n_edges, num_nodes, dtype=torch.float, device=edge_idx.device)

        # Create incident matrix
        edges_to_nodes[torch.arange(n_edges), edge_idx[0]] = 1
        edges_to_nodes[torch.arange(n_edges), edge_idx[1]] = 1

        # Compute the dual adjacency matrix using matrix multiplication
        dual_adj = torch.matmul(edges_to_nodes, edges_to_nodes.T)

        # Set the diagonal elements to zero
        dual_adj.fill_diagonal_(0)

        # Get the indices of non-zero elements, which represent the dual edges
        dual_edge_idx = torch.nonzero(dual_adj, as_tuple=False).T

        return dual_edge_idx
        
    
    def forward(self, node_feats, edge_idx):
        # Create edge features by concatenating node embeddings
        edge_feats = torch.cat([node_feats[edge_idx[0]], node_feats[edge_idx[1]]], dim=1)
        dual_edge_idx = self.dual_edge_idx(edge_idx, node_feats.shape[0])
        edge_feats = self.gin(edge_feats, dual_edge_idx).squeeze(1)

        return edge_feats