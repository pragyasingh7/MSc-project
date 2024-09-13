import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm

from torch_geometric.utils import dense_to_sparse


def load_bi_sr_model(config, target_node_embeddings):
    if config.model.sr_method == 'linear_algebraic':
        model = LA(config.data.n_source_nodes, 
                         config.data.n_target_nodes,
                         num_heads=config.model.num_heads, 
                         edge_dim=config.model.edge_dim,
                         dropout=config.model.dropout, 
                         beta=config.model.beta,
                         min_max_scale=config.model.min_max_scale,
                         multi_dim_edge=config.model.multi_dim_edge,
                         binarize=config.model.binarize)
        
    elif config.model.sr_method == 'bi_lc':
        model = BiLC(config.data.n_source_nodes, 
                               config.data.n_target_nodes,
                               num_heads=config.model.num_heads, 
                               edge_dim=config.model.edge_dim,
                               dropout=config.model.dropout, 
                               beta=config.model.beta,
                               hidden_dim=config.model.hidden_dim, 
                               refine_target=config.model.refine_target,
                               target_refine_fully_connected=config.model.target_refine_fully_connected,
                               min_max_scale=config.model.min_max_scale,
                               multi_dim_edge=config.model.multi_dim_edge,
                               binarize=config.model.binarize)
        
    elif config.model.sr_method == 'bi_mp':
        model = BiMP(config.data.n_source_nodes, 
                             config.data.n_target_nodes,
                             target_node_embeddings,
                             num_heads=config.model.num_heads, 
                             edge_dim=config.model.edge_dim,
                             dropout=config.model.dropout, 
                             beta=config.model.beta,
                             hidden_dim=config.model.hidden_dim, 
                             refine_target=config.model.refine_target,
                             target_refine_fully_connected=config.model.target_refine_fully_connected,
                             min_max_scale=config.model.min_max_scale,
                             multi_dim_edge=config.model.multi_dim_edge,
                             binarize=config.model.binarize) 
        
    else:
        raise ValueError(f"Invalid SR method: {config.model.sr_method}")

    return model   
    

class LA(nn.Module):
    """TransformerConv based node LA method"""
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1, 
                 dropout=0.2, beta=False, min_max_scale=True, multi_dim_edge=False,
                 binarize=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0

        self.min_max_scale = min_max_scale
        self.multi_dim_edge = multi_dim_edge
        self.binarize = binarize

        self.conv1 = TransformerConv(n_source_nodes, n_target_nodes // num_heads, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(n_target_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)                               # (n_s, n_t)

        if self.multi_dim_edge:
            # Compute edge features for the target graph using sum of incident node features
            xt = x.unsqueeze(1) + x.unsqueeze(2)    # (n_s, n_t, n_t)
            xt  =xt.permute(1, 2, 0)                # (n_t, n_t, n_s)

        else:
            # Compute edge values using dot product
            xt = x.T @ x                            # (n_t, n_t)

            # Normalize values to be between [0, 1]
            if self.min_max_scale:
                xt_min = torch.min(xt)
                xt_max = torch.max(xt)
                xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)

            if self.binarize:
                # add z-normalization
                xt = (xt - torch.mean(xt)) / torch.std(xt)
                # binarize the matrix through sigmoid
                xt = torch.sigmoid(xt)

        return xt
    

class BiLC(nn.Module):
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1, 
                 dropout=0.2, beta=False, hidden_dim=32, refine_target=False,
                 target_refine_fully_connected=False, min_max_scale=True, 
                 multi_dim_edge=False, binarize=False):
        super().__init__()
        self.n_target_nodes = n_target_nodes

        self.min_max_scale = min_max_scale
        self.multi_dim_edge = multi_dim_edge
        self.binarize = binarize

        self.conv1 = TransformerConv(n_source_nodes, hidden_dim, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(hidden_dim * num_heads)

        self.bipartitie_fc = nn.Linear(n_source_nodes, n_target_nodes)

        self.refine_target = refine_target
        self.target_refine_fully_connected = target_refine_fully_connected
        if refine_target:
            # Refine target adjacency matrix
            if not target_refine_fully_connected:
                self.target_adj_fc = nn.Linear(n_source_nodes, n_target_nodes)
                edge_dim = 1
            else:
                edge_dim = None
        
            self.conv2 = TransformerConv(hidden_dim * num_heads, hidden_dim, 
                                         heads=num_heads, edge_dim=edge_dim,
                                         dropout=dropout, beta=beta)
            self.bn2 = GraphNorm(hidden_dim * num_heads)

    def create_fully_connected_edge_index(self):
        num_nodes = self.n_target_nodes
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index


    def linear_sr(self, x):
        xt = self.bipartitie_fc(x.T).T              # (n_t, h)

        # Refine target nodes
        if self.refine_target:
            if self.target_refine_fully_connected:
                # Assume a fully connected target graph
                edge_index = self.create_fully_connected_edge_index().to(x.device)
                xt = self.conv2(xt, edge_index)     # (n_t, h)
            else:
                # Learn target graph domain, inspired from GiG paper
                xt_refine_domain = self.target_adj_fc(x.T).T      # (n_t, h)
                # xt_refine_domain = F.relu(xt_refine_domain)     

                # Take dot product followed by sigmoid to get adjacency matrix
                adj_t_refine = xt_refine_domain @ xt_refine_domain.T
                adj_t_refine = F.sigmoid(adj_t_refine)

                # Apply thresholding on the adjacency matrix
                adj_mask = (adj_t_refine > 0.5).float()
                adj_masked = adj_t_refine * adj_mask

                edge_index, edge_attr = dense_to_sparse(adj_masked)

                xt = self.conv2(xt, edge_index, edge_attr.view(-1, 1))

            xt = self.bn2(xt)
            xt = F.relu(xt)                        # (n_t, h)

        if self.multi_dim_edge:
            # Compute edge features for the target graph using sum of incident node features
            xt = xt.unsqueeze(0) + xt.unsqueeze(1)  # (n_t, n_t, h)

        else:      
            # Take dot product to get (n_t, nt)
            xt = xt @ xt.T                       # (n_t, n_t)
        
        return xt


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        xt = self.linear_sr(x)  # (n_t, n_t) or (n_t, n_t, h)

        if not self.multi_dim_edge:
            # Normalize values to be between [0, 1]
            if self.min_max_scale:
                xt_min = torch.min(xt)
                xt_max = torch.max(xt)
                xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

            if self.binarize:
                # add z-normalization
                xt = (xt - torch.mean(xt)) / torch.std(xt)
                # binarize the matrix through sigmoid
                xt = torch.sigmoid(xt)

        return xt


class BiMP(nn.Module):
    def __init__(self, n_source_nodes, n_target_nodes, target_node_embeddings, num_heads=4, edge_dim=1, 
                 dropout=0.2, beta=False, hidden_dim=32, refine_target=False,
                 target_refine_fully_connected=True, min_max_scale=True,
                 multi_dim_edge=False, binarize=False):
        super().__init__()
        assert target_node_embeddings is not None
        self.n_target_nodes = n_target_nodes
        self.n_source_nodes = n_source_nodes

        self.min_max_scale = min_max_scale
        self.multi_dim_edge = multi_dim_edge
        self.binarize = binarize

        self.conv1 = TransformerConv(n_source_nodes, hidden_dim, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(hidden_dim * num_heads)

        # Fixed target node embedding to specify target node order
        self.target_node_embeddings = target_node_embeddings
        
        self.bipartite_attn = TransformerConv(hidden_dim * num_heads, hidden_dim, 
                                              heads=num_heads,
                                              dropout=dropout, beta=beta)
        self.bipartite_attn_bn = GraphNorm(hidden_dim * num_heads)
        
        self.refine_target = refine_target
        self.target_refine_fully_connected = target_refine_fully_connected
        if refine_target:
            # Refine target adjacency matrix
            if not target_refine_fully_connected:
                self.conv3 = TransformerConv(hidden_dim * num_heads, hidden_dim,
                                                heads=num_heads,
                                                dropout=dropout, beta=beta)
                self.bn3 = GraphNorm(hidden_dim * num_heads)
                edge_dim = 1
            else:
                edge_dim = None

            # Refine target nodes
            self.conv2 = TransformerConv(hidden_dim * num_heads, hidden_dim, 
                                         heads=num_heads, edge_dim=edge_dim,
                                         dropout=dropout, beta=beta)
            self.bn2 = GraphNorm(hidden_dim * num_heads)


    def create_fully_connected_edge_index(self):
        num_nodes = self.n_target_nodes
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index
    
    def create_bipartite_edge_index(self):
        n_s = self.n_source_nodes
        n_t = self.n_target_nodes
        # Group 1 node indices (0 to n_s-1)
        group1 = torch.arange(n_s)
        # Group 2 node indices (n_s to n_s + n_t - 1)
        group2 = torch.arange(n_s, n_s + n_t)

        # Create all possible edges between group 1 and group 2
        row = group1.repeat_interleave(n_t)
        col = group2.repeat(n_s)

        # Stack the row and col tensors to form edge_index
        edge_index = torch.stack([row, col], dim=0)

        return edge_index

    def attn_sr(self, x):
        # Initialize the bipartite graph's node features 
        x_bipartite = torch.cat([x, self.target_node_embeddings], dim=0)    # (n_s + n_t, h)

        # Intialize the edge indices
        edge_index_bipartite = self.create_bipartite_edge_index().to(x.device)

        # Apply TransformerConv to get the target node embeddings
        xt = self.bipartite_attn(x_bipartite, edge_index_bipartite)  # (n_t + n_s, h)
        xt = self.bipartite_attn_bn(xt)
        xt = F.relu(xt)
        # Fetch the target node embeddings
        xt = xt[self.n_source_nodes:]               # (n_t, h)

        # Refine target nodes
        if self.refine_target:
            if self.target_refine_fully_connected:
                # Assume a fully connected target graph
                edge_index = self.create_fully_connected_edge_index().to(x.device)
                xt = self.conv2(xt, edge_index)     # (n_t, h)
            else:
                # Learn target graph domain, inspired from GiG paper
                x_bipartite_adj = torch.cat([x, self.target_node_embeddings], dim=0)    # (n_s + n_t, h)
                xt_refine_domain = self.conv3(x_bipartite_adj, edge_index_bipartite)
                xt_refine_domain = self.bn3(xt_refine_domain)      # (n_t + n_s, h)
                # xt_refine_domain = F.relu(xt_refine_domain)     

                # Fetch target node embeddings
                xt_refine_domain = xt_refine_domain[self.n_source_nodes:]  # (n_t, h)

                # Take dot product followed by sigmoid to get adjacency matrix
                adj_t_refine = xt_refine_domain @ xt_refine_domain.T
                adj_t_refine = F.sigmoid(adj_t_refine)

                # Apply thresholding on the adjacency matrix
                adj_mask = (adj_t_refine > 0.5).float()
                adj_masked = adj_t_refine * adj_mask

                edge_index, edge_attr = dense_to_sparse(adj_masked)

                xt = self.conv2(xt, edge_index, edge_attr.view(-1, 1))

            xt = self.bn2(xt)
            xt = F.relu(xt)                        # (n_t, h)

        if self.multi_dim_edge:
            # Compute edge features for the target graph using sum of incident node features
            xt = xt.unsqueeze(0) + xt.unsqueeze(1)  # (n_t, n_t, h)
        else:
            # Take dot product to get (n_t, nt)
            xt = xt @ xt.T

        return xt
    

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        xt = self.attn_sr(x)  # (n_t, n_t) or (n_t, n_t, h)

        # Normalize values to be between [0, 1]
        if not self.multi_dim_edge:
            if self.min_max_scale:
                xt_min = torch.min(xt)
                xt_max = torch.max(xt)
                xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

            if self.binarize:
                # add z-normalization
                xt = (xt - torch.mean(xt)) / torch.std(xt)
                # binarize the matrix through sigmoid
                xt = torch.sigmoid(xt)

        return xt