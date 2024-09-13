import torch
import numpy as np
import networkx as nx
from functools import partial
from data.utils import create_pyg_graph, graph_to_pyg_nodes, graph_to_node_feat

from data.topo_metric import topK_subgraph


def load_dataset(config, device):
    """Load the training dataset."""
    # Dataset configuration
    n_target_nodes = config.data.n_target_nodes
    reduction_metric = config.data.reduction_metric
    n_source_nodes = config.data.n_source_nodes

    m_min = config.data.m_min
    m_max = config.data.m_max

    n_samples = config.data.n_samples

    # Create random SBM graphs
    target_graphs = create_random_ba_graphs(n_samples, m_min, m_max, n_target_nodes)
    
    source_graphs = [topK_subgraph(G, reduction_metric, n_source_nodes) for G in target_graphs]

    # # Get node features
    node_feat_partial = partial(graph_to_node_feat, 
                                feat_type=config.data.node_feat_type,
                                x_dim=config.data.node_feat_dim,
                                n2v_walk_length=config.data.n2v_walk_length,
                                n2v_num_walks=config.data.n2v_num_walks,
                                device=device)
    source_node_feat_all = [node_feat_partial(G) for G in source_graphs]
    target_node_feat_all = [node_feat_partial(G) for G in target_graphs]

    # Get adj matrix
    source_mat_all = [torch.corrcoef(x) for x in source_node_feat_all]
    target_mat_all = [torch.corrcoef(x) for x in target_node_feat_all]

    # Mask out self-loops
    self_loop_mask_source = 1 - torch.eye(n_source_nodes, device=device)
    self_loop_mask_target = 1 - torch.eye(n_target_nodes, device=device)

    source_mat_all = [x * self_loop_mask_source for x in source_mat_all]
    target_mat_all = [x * self_loop_mask_target for x in target_mat_all]

    # Convert to PyG graph
    pyg_partial = partial(create_pyg_graph, node_feature_init=config.data.node_feat_init, 
                          node_feat_dim=config.data.node_feat_dim)

    source_pyg_all = [pyg_partial(x, n_source_nodes) for x in source_mat_all]
    target_pyg_all = [pyg_partial(x, n_target_nodes) for x in target_mat_all]

    return source_pyg_all, target_pyg_all, source_mat_all, target_mat_all


def load_gunet_dataset(config, device):
    """Load training dataset to super-resolve nodes w/o edge prediction"""
    # Dataset configuration
    n_target_nodes = config.data.n_target_nodes

    m_min = config.data.m_min
    m_max = config.data.m_max

    n_samples = config.data.n_samples

    # Create random SBM graphs
    target_graphs = create_random_ba_graphs(n_samples, m_min, m_max, n_target_nodes)
    
    target_pyg_all = [graph_to_pyg_nodes(G, feat_type=config.data.node_feat_init, device=device) for G in target_graphs]

    return target_pyg_all


def create_random_ba_graphs(
        n_samples, 
        m_min,
        m_max,
        n_nodes = 100):
    
    graphs = []

    for _ in range(n_samples):
        m = np.random.randint(m_min, m_max)
        G = nx.barabasi_albert_graph(n_nodes, m)
        graphs.append(G)

    return graphs


