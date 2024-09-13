import copy
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from models.bi_sr import load_bi_sr_model
from data.utils import create_pyg_graph


def load_ae_model(config_lr, n_source_nodes, n_target_nodes, target_node_embeddings=None):

    model = AutoEncoder(config_lr, n_source_nodes, n_target_nodes, target_node_embeddings)

    return model   


class AutoEncoder(nn.Module):
    def __init__(self, config_lr, n_source_nodes, n_target_nodes, target_node_embeddings=None):
        super().__init__()

        config_lr.data.n_source_nodes = n_source_nodes
        config_lr.data.n_target_nodes = n_target_nodes
        self.lr_encoder = load_bi_sr_model(config_lr, target_node_embeddings)

        config_hr = copy.deepcopy(config_lr)
        config_hr.data.n_source_nodes = n_target_nodes
        config_hr.data.n_target_nodes = n_source_nodes
        self.hr_encoder = load_bi_sr_model(config_hr, target_node_embeddings)

        self.create_pyg_graph = partial(create_pyg_graph, node_feature_init='adj')

    def forward(self, source_data):
        hr_adj = self.lr_encoder(source_data)
        hr_pyg = self.create_pyg_graph(hr_adj, hr_adj.shape[0])
        lr_adj = self.hr_encoder(hr_pyg)

        return lr_adj, hr_adj