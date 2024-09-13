import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm

from models.bi_sr import load_bi_sr_model


def load_dual_model(config, target_node_embeddings):
    dual_node_in_dim = 1
    dual_binarize = config.model.binarize

    if config.model.multi_dim_edge:
        if config.model.sr_method == 'linear_algebraic':
            dual_node_in_dim = config.data.n_source_nodes

        elif config.model.sr_method in ['bi_lc', 'bi_mp']:
            dual_node_in_dim = config.model.hidden_dim * config.model.num_heads
        
        else:
            raise ValueError(f"Invalid SR method: {config.model.sr_method}")
    
    # If binarize is True, then target_node_init model should get binarize=False
    if dual_binarize:
        config.model.binarize = False

    model = DualModel(config, 
                      target_node_embeddings,
                      config.data.n_target_nodes, 
                      dual_node_in_dim=dual_node_in_dim, 
                      dual_node_out_dim=config.model.dual_node_out_dim, 
                      num_heads=config.model.num_heads, 
                      drouput=config.model.dropout, 
                      beta=config.model.beta, 
                      min_max_scale=config.model.min_max_scale,
                      binarize=dual_binarize)

    return model
    

class DualLearner(nn.Module):
    def __init__(self, in_dim, out_dim=1, num_heads=4, 
                 dropout=0.2, beta=False, min_max_scale=True, binarize=False):
        super().__init__()
        if out_dim == 1:
            num_heads = 1
        else:
            assert out_dim % num_heads == 0

        self.min_max_scale = min_max_scale
        self.binarize = binarize

        self.conv1 = TransformerConv(in_dim, out_dim // num_heads, 
                                     heads=num_heads,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(out_dim)


    def forward(self, x, edge_index):
        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        xt = F.relu(x)              # (n_t*(n_t-1)/2, h)

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
    

class DualModel(nn.Module):
    def __init__(self, config, target_node_embeddings, n_target_nodes, dual_node_in_dim=1, 
                 dual_node_out_dim=1, num_heads=4, drouput=0.2, beta=False, min_max_scale=True,
                 binarize=False):
        super().__init__()
        self.n_target_nodes = n_target_nodes
        self.n_ut = n_target_nodes * (n_target_nodes - 1) // 2

        self.node_init_model = load_bi_sr_model(config, target_node_embeddings)

        self.dual_learner = DualLearner(dual_node_in_dim, dual_node_out_dim, 
                                        num_heads, drouput, beta, min_max_scale,
                                        binarize)

    def get_dual_node_init(self, source_data):
        dual_node_init_x = self.node_init_model(source_data)    # (n_t, n_t) or (n_t, n_t, h)
        
        # Fetch the upper triangular part of the matrix
        ut_mask = torch.triu(torch.ones(self.n_target_nodes, self.n_target_nodes), diagonal=1).bool().to(dual_node_init_x.device)
        if len(dual_node_init_x.shape) == 3:
            ut_mask = ut_mask.unsqueeze(2).expand(-1, -1, dual_node_init_x.shape[2])
        # ut_mask = torch.triu(torch.ones_like(dual_node_init_x), diagonal=1).bool()
        dual_x = torch.masked_select(dual_node_init_x, ut_mask)
        dual_x = dual_x.view(self.n_ut, -1)     # (n_t*(n_t-1)/2, 1) or (n_t*(n_t-1)/2, h)
        
        return dual_x
    
    def forward(self, source_data, dual_edge_index):
        # Update target node embeddings
        dual_node_init = self.get_dual_node_init(source_data)   # (n_t*(n_t-1)/2, 1) or (n_t*(n_t-1)/2, h)
        
        dual_target_x = self.dual_learner(dual_node_init, dual_edge_index)

        return dual_target_x