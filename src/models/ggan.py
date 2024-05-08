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
       
        self.conv21 = GCNConv(n_source_nodes, 2 * n_target_nodes, cached=cached)
        self.conv211 = BatchNorm(2 * n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)
        self.conv22 = GCNConv(2 * n_target_nodes, n_target_nodes, cached=cached)
        self.conv222 = BatchNorm(n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)
        self.conv23 = GCNConv(n_target_nodes, n_target_nodes, cached=cached)
      

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = torch.squeeze(x)
        x = self.conv21(x, edge_index).relu()
        x1 = F.sigmoid(self.conv211(x))
        x1 = F.dropout(x1, training=self.training)
       
        x2 = self.conv22(x1, edge_index).relu()
        x2 = F.sigmoid(self.conv222(x2))
        x2 = F.dropout(x2, training=self.training)
       
        x3  = (torch.matmul(x2.t(), x2)) 

        return x3


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
        
        self.conv21 = GCNConv(n_target_nodes, n_target_nodes, cached=cached)
        self.conv211 = BatchNorm(n_target_nodes, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
        self.conv211 = BatchNorm(n_target_nodes, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)
        self.conv22 = GCNConv(n_target_nodes, 1, cached=cached)
        self.conv222 = BatchNorm(1, 
                                eps=bn_eps, 
                                momentum=bn_momentum, 
                                affine=bn_affine, 
                                track_running_stats=bn_track_running_stats)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x = torch.squeeze(x)
        x = self.conv21(x, edge_index).relu()
        x = self.conv211(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x1 = F.relu(self.conv222(self.conv22(x, edge_index)))

        return F.sigmoid(x1)
