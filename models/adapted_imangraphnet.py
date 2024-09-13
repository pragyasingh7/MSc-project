import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, GCNConv, BatchNorm

from data_utils import create_pyg_graph, MatrixVectorizer


def load_iman_model(config):
    model = AdaptedIMANGraphNet(config.data.n_source_nodes, 
                                config.data.n_target_nodes, 
                                hidden_dim=config.model.hidden_dim)
    return model


class Aligner(nn.Module):
    def __init__(self, n_source_nodes, hidden_dim=32):     
        super().__init__()

        self.n_source_nodes = n_source_nodes
        self.proj_in = nn.Linear(n_source_nodes, hidden_dim)
        # self.proj_in = nn.Sequential(nn.Linear(n_source_nodes, hidden_dim), nn.ReLU())
        
        fc = nn.Sequential(nn.Linear(1, hidden_dim * 2 * hidden_dim), nn.ReLU())
        self.conv1 = NNConv(hidden_dim, 2*hidden_dim, fc, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(2*hidden_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        fc = nn.Sequential(nn.Linear(1, 2*hidden_dim), nn.ReLU())
        self.conv2 = NNConv(2*hidden_dim, 1, fc, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.conv3 = NNConv(1, hidden_dim, fc, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(hidden_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        self.proj_out1 = nn.Linear(hidden_dim, n_source_nodes)
        self.proj_out2 = nn.Linear(2*hidden_dim, n_source_nodes)
        # self.proj_out1 = nn.Sequential(nn.Linear(hidden_dim, n_source_nodes), nn.ReLU())
        # self.proj_out2 = nn.Sequential(nn.Linear(2*hidden_dim, n_source_nodes), nn.ReLU())

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x = F.sigmoid(self.proj_in(x))

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr)))
        x3 = self.proj_out1(x3)

        x1_ = self.proj_out2(x1)

        x3 = F.sigmoid(torch.cat([x3, x1_], dim=1))
        x4 = x3[:, 0:self.n_source_nodes]
        x5 = x3[:, self.n_source_nodes:2*self.n_source_nodes]

        x6 = (x4 + x5) / 2
        return x6


class Generator(nn.Module):
    def __init__(self, n_source_nodes, n_target_nodes, hidden_dim=32):
        super().__init__()

        self.proj_in = nn.Linear(n_source_nodes, hidden_dim)
        # self.proj_in = nn.Sequential(nn.Linear(n_source_nodes, hidden_dim), nn.ReLU())
        
        fc = nn.Sequential(nn.Linear(1, hidden_dim * 2 * hidden_dim), nn.ReLU())
        self.conv1 = NNConv(hidden_dim, 2*hidden_dim, fc, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(2*hidden_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        fc = nn.Sequential(nn.Linear(1, 2*hidden_dim*4*hidden_dim), nn.ReLU())
        self.conv3 = NNConv(2*hidden_dim, 4*hidden_dim, fc, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(4*hidden_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        self.proj_out = nn.Linear(4*hidden_dim, n_target_nodes)
        # self.proj_out = nn.Sequential(nn.Linear(4*hidden_dim, n_target_nodes), nn.ReLU())


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        # x = torch.squeeze(x)

        x = F.sigmoid(self.proj_in(x))
        
        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x1, edge_index, edge_attr)))
        x3 = F.dropout(x3, training=self.training)

        x3 = F.sigmoid(self.proj_out(x3))

        x4  = torch.matmul(x3.t(), x3)

        return x4

class Discriminator(nn.Module):
    def __init__(self, n_target_nodes):
        super().__init__()
        self.conv1 = GCNConv(n_target_nodes, n_target_nodes, cached=True)
        self.conv2 = GCNConv(n_target_nodes, 1, cached=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = torch.squeeze(x)
        x1 = F.sigmoid(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x2 = F.sigmoid(self.conv2(x1, edge_index))

        return x2
    

class AdaptedIMANGraphNet(nn.Module):
    def __init__(self, n_source_nodes, n_target_nodes, hidden_dim=32):
        super().__init__()

        self.n_source_nodes = n_source_nodes
        self.n_target_nodes = n_target_nodes
        
        self.n_source_nodes_f = (n_source_nodes * (n_source_nodes - 1)) // 2

        self.aligner = Aligner(n_source_nodes, hidden_dim)
        self.generator = Generator(n_source_nodes, n_target_nodes, hidden_dim)
        self.discriminator = Discriminator(n_target_nodes)

        self.create_pyg_graph = partial(create_pyg_graph, node_feature_init='adj')
        self.matrix_vec = MatrixVectorizer()

    def kl_loss(self, target, predicted):
        kl_loss = torch.abs(F.kl_div(F.softmax(target, dim=-1), F.softmax(predicted, dim=-1), None, None, 'sum'))
        kl_loss = (1/350) * kl_loss
        return kl_loss
    
    def align_loss(self, target_data, a_output):
        target = target_data.edge_attr.view(self.n_target_nodes, -1).detach().clone()
        target_mean = torch.mean(target)
        target_std = torch.std(target)

        d_target = torch.normal(target_mean, target_std, size=(self.n_source_nodes_f,))
        target_d = torch.tensor(self.matrix_vec.anti_vectorize(d_target, self.n_source_nodes)).to(a_output.device)

        loss = self.kl_loss(target_d, a_output)
        return loss

    def dis_loss(self, d_real, d_fake):
        loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real, requires_grad=True))
        loss_fake = F.binary_cross_entropy(d_fake.detach(), torch.zeros_like(d_fake, requires_grad=True))
        return (loss_real + loss_fake) / 2
    
    def gen_loss(self, target_data, g_output, d_fake):
        target = target_data.edge_attr.view(self.n_target_nodes, -1)
        loss1 = F.l1_loss(target, g_output)
        loss2 = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake, requires_grad=True))
        return loss1 + loss2, loss1

    def forward(self, source_data, target_data):
        # Aligner output
        a_output = self.aligner(source_data)
        a_casted = self.create_pyg_graph(a_output, self.n_source_nodes)

        # Generator output
        g_output = self.generator(a_casted)

        # Create discriminator input graph
        g_output_casted = self.create_pyg_graph(g_output.detach(), self.n_target_nodes)

        # Discriminator output
        d_real = self.discriminator(target_data)
        d_fake = self.discriminator(g_output_casted)

        # Losses
        al_loss = self.align_loss(target_data, a_output)
        d_loss = self.dis_loss(d_real, d_fake)
        g_loss, g_loss_l1 = self.gen_loss(target_data, g_output, d_fake)

        return g_output, al_loss, g_loss, g_loss_l1, d_loss
    
