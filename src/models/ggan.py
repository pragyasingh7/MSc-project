import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm


class GCNGenerator(nn.Module):
    def __init__(
                self,
                n_source_nodes,
                n_target_nodes,
                cached=True,
                bn_eps=1e-03,
                bn_momentum=0.1,
                bn_affine=True,
                bn_track_running_stats=True
    ):
        super().__init__()
       
        self.conv11 = GCNConv(n_source_nodes, 2 * n_target_nodes, cached=cached)
        self.conv12 = BatchNorm(2 * n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)

        self.conv21 = GCNConv(2 * n_target_nodes, 4 * n_target_nodes, cached=cached)
        self.conv22 = BatchNorm(4 * n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)

        self.conv31 = GCNConv(4 * n_target_nodes, 2 * n_target_nodes, cached=cached)
        self.conv32 = BatchNorm(2 * n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)

        self.conv41 = GCNConv(2 * n_target_nodes, n_target_nodes, cached=cached)
        self.conv42 = BatchNorm(n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)
        
        self.conv51 = GCNConv(n_target_nodes, n_target_nodes, cached=cached)


    def enforce_priors(self, x):
        # Ensure symmetry
        x1 = (x + x.t()) / 2

        # Ensure no self-loops
        x2 = x1 - torch.diag(torch.diag(x1))

        return x2

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        # x = torch.squeeze(x)

        x1 = F.sigmoid(self.conv12(self.conv11(x, edge_index, edge_attr)))
        x2 = F.dropout(x1, training=self.training)

        x3 = F.sigmoid(self.conv22(self.conv21(x2, edge_index, edge_attr)))
        x4 = F.dropout(x3, training=self.training)

        x5 = F.sigmoid(self.conv32(self.conv31(x4, edge_index, edge_attr)))
        x6 = F.dropout(x5, training=self.training)

        x7 = F.sigmoid(self.conv42(self.conv41(x6, edge_index, edge_attr)))
        x8 = F.dropout(x7, training=self.training)

        x9 = F.sigmoid(self.conv51(x8, edge_index, edge_attr))
        x10 = F.dropout(x9, training=self.training)

        x11  = torch.matmul(x10.t(), x10)

        x12 = self.enforce_priors(x11)

        return x12


class GCNDiscriminator(torch.nn.Module):
    def __init__(
                self,
                n_source_nodes,
                n_target_nodes,
                cached=True,
                bn_eps=1e-03,
                bn_momentum=0.1,
                bn_affine=True,
                bn_track_running_stats=True
    ):
        super().__init__()
        
        self.conv1 = GCNConv(n_target_nodes, n_target_nodes, cached=cached)
        # self.conv11 = BatchNorm(2 * n_target_nodes, 
        #                         eps=bn_eps, 
        #                         momentum=bn_momentum, 
        #                         affine=bn_affine, 
        #                         track_running_stats=bn_track_running_stats)
        # self.conv2 = GCNConv(n_target_nodes, n_target_nodes, cached=cached)
        self.nn = nn.Linear(n_target_nodes, 1)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # x = torch.squeeze(x)

        # x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index)))
        x1 = F.sigmoid(self.conv1(x, edge_index))
        x2 = F.dropout(x1, training=self.training)

        # x2 = F.sigmoid(self.conv2(x1, edge_index))

        # Do graph-level prediction
        x3 = torch.mean(x2, dim=0)  # Global mean pooling
        x4 = F.sigmoid(self.nn(x3))

        return x4

