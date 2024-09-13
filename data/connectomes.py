import torch
from functools import partial
import pandas as pd
from data_utils import create_pyg_graph, MatrixVectorizer


def load_dataset(config, device):
    """
    Load the dataset for training the model.
    """
    # Load dataset
    n_source_nodes = config.data.n_source_nodes
    n_target_nodes = config.data.n_target_nodes

    source_dir = config.data.source_dir
    target_dir = config.data.target_dir

    source_data = pd.read_csv(source_dir).to_numpy()
    target_data = pd.read_csv(target_dir).to_numpy()

    # Convert to matrix
    matrix_vec = MatrixVectorizer()
    source_mat_all = [matrix_vec.anti_vectorize(x, n_source_nodes) for x in source_data]
    target_mat_all = [matrix_vec.anti_vectorize(x, n_target_nodes) for x in target_data]

    # Convert to torch tensors
    source_mat_all = [torch.tensor(x, dtype=torch.float, device=device) for x in source_mat_all]
    target_mat_all = [torch.tensor(x, dtype=torch.float, device=device) for x in target_mat_all]

    # Convert to PyG graph
    pyg_partial = partial(create_pyg_graph, node_feature_init=config.data.node_feat_init, 
                          node_feat_dim=config.data.node_feat_dim)

    source_pyg_all = [pyg_partial(x, n_source_nodes) for x in source_mat_all]
    target_pyg_all = [pyg_partial(x, n_target_nodes) for x in target_mat_all]

    return source_pyg_all, target_pyg_all, source_mat_all, target_mat_all