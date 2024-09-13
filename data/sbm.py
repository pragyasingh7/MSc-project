import torch
import numpy as np
import networkx as nx
from functools import partial
from data.utils import create_pyg_graph, graph_to_pyg_nodes, graph_to_node_feat, create_pyg_simulated_nodes

from data.topo_metric import topK_subgraph


def load_dataset(config, device):
    """Load the training dataset."""
    # Dataset configuration
    n_target_nodes = config.data.n_target_nodes
    reduction_metric = config.data.reduction_metric
    n_source_nodes = config.data.n_source_nodes

    n_blocks_min = config.data.n_blocks_min
    n_blocks_max = config.data.n_blocks_max
    p_inter_min = config.data.p_inter_min
    p_inter_max = config.data.p_inter_max
    p_intra_min = config.data.p_intra_min
    p_intra_max = config.data.p_intra_max

    n_samples = config.data.n_samples

    # Create random SBM graphs
    target_graphs = create_random_sbm_graphs(n_samples, n_blocks_min, n_blocks_max, 
                                         p_inter_min, p_inter_max, p_intra_min, 
                                         p_intra_max, n_target_nodes)
    
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

    # Convert to torch tensors
    source_mat_all = [torch.tensor(x, dtype=torch.float, device=device) for x in source_mat_all]
    target_mat_all = [torch.tensor(x, dtype=torch.float, device=device) for x in target_mat_all]

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

    n_blocks_min = config.data.n_blocks_min
    n_blocks_max = config.data.n_blocks_max
    p_inter_min = config.data.p_inter_min
    p_inter_max = config.data.p_inter_max
    p_intra_min = config.data.p_intra_min
    p_intra_max = config.data.p_intra_max

    n_samples = config.data.n_samples

    # Create random SBM graphs
    target_graphs = create_random_sbm_graphs(n_samples, n_blocks_min, n_blocks_max, 
                                         p_inter_min, p_inter_max, p_intra_min, 
                                         p_intra_max, n_target_nodes)
    
    target_pyg_all = [graph_to_pyg_nodes(G, feat_type=config.data.node_feat_init, device=device) for G in target_graphs]

    return target_pyg_all


def create_prob_matrix(n_blocks, p_inter_min, p_inter_max, p_intra_min, p_intra_max):
    # Create a matrix of random values within the inter-block range
    p = torch.rand(n_blocks, n_blocks) * (p_inter_max - p_inter_min) + p_inter_min

    # Generate intra-block random values
    intra_probs = torch.rand(n_blocks) * (p_intra_max - p_intra_min) + p_intra_min

    # Make p symmetric
    p = (p + p.t()) / 2

    # Create a diagonal matrix from intra_probs
    diag = torch.diag(intra_probs)

    # Combine the matrices: diagonal for intra-block and off-diagonal for inter-block
    # Use a mask to zero out the original diagonal in 'p' and then add the 'diag' matrix
    mask = torch.eye(n_blocks, dtype=torch.bool)
    p = p.masked_fill(mask, 0) + diag

    return p

def random_partition(num_nodes, num_blocks):
    partitions = np.random.multinomial(num_nodes - num_blocks, np.ones(num_blocks) / num_blocks) + 1
    return partitions

def create_random_sbm_graphs(
        n_samples, 
        n_blocks_min, 
        n_blocks_max,  
        p_inter_min, 
        p_inter_max, 
        p_intra_min, 
        p_intra_max,
        n_nodes = 100):
    
    graphs = []

    for i in range(n_samples):
        # Select number of 
        n_blocks = np.random.randint(n_blocks_min, n_blocks_max)
        # n_nodes_per_block = np.random.randint(n_nodes_per_block_min, 
        #                                       n_nodes_per_block_max,
        #                                       n_blocks)
        n_nodes_per_block = random_partition(n_nodes, n_blocks)
        # Create probability matrix
        p = create_prob_matrix(n_blocks, p_inter_min, p_inter_max, p_intra_min, p_intra_max)

        G = nx.stochastic_block_model(n_nodes_per_block, p)

        graphs.append(G)

    return graphs


