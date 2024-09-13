DATASET_TYPE = 'sbm'  # 'sbm', 'ba', 'ws'

class BaseConfig:
    class experiment:
        # Parameters for K-fold cross validation
        num_splits = 3
        shuffle = True
        random_state = 42

        # Training parameters
        max_epochs = 150
        lr = 0.001
        batch_size = 16

        # Early-stopping parameters
        early_stopping = True
        patience = 5
        warm_up_epochs = 15

        # Logging results
        base_dir = 'results'
        run_name = 'simulated'

    class model:
        sr_method = 'matrix_multiplication'  # 'matrix_multiplication', 'linear_combination', 'attention'
        use_dual = False    # Whether to use the dual graph formulation

        # Whether to refine learned target node features (for 'linear_combination' and 'attention')
        refine_target = False

        # Whether to use fully connected or learn adj matrix used to refine target node features (inspired by GiG, for 'linear_combination' and 'attention')
        target_refine_fully_connected = True

        # TransformerConv parameters
        hidden_dim = 32
        num_heads = 4
        dropout = 0.2
        edge_dim = 1
        beta = False

        # If mix max scaling should be applied to output. Generally, used when multi_dim_edge is False
        min_max_scale = True

        # If output type is binary. In that case, use sigmoid non-linearity
        binarize = False

        # If output matrix from node GTB is multi-dimensional i.e. (n_t, n_t, h)
        multi_dim_edge = False

        # Parameters for the Dual Model
        dual_node_in_dim = 1
        dual_node_out_dim = 1

class SBMConfig(BaseConfig):
    class data:
        type = 'sbm'
        load_data = False
        n_target_nodes = 64
        reduction_ratio = 0.5
        reduction_metric = 'degree'
        n_source_nodes = int(n_target_nodes * reduction_ratio)

        n_samples = 128

        n_blocks_min = 2
        n_blocks_max = 5
        p_inter_min = 0.01
        p_inter_max = 0.10
        p_intra_min = 0.5
        p_intra_max = 0.6

        node_feat_init = 'adj'
        node_feat_dim = 8
        node_feat_type = 'node2vec'
        n2v_walk_length = 0.8
        n2v_num_walks = 100


class BAConfig(BaseConfig):
    class data:
        type = 'ba'
        load_data = False
        n_target_nodes = 64
        reduction_ratio = 0.5
        reduction_metric = 'degree'
        n_source_nodes = int(n_target_nodes * reduction_ratio)

        n_samples = 128

        m_min = 4
        m_max = 8

        node_feat_init = 'adj'
        node_feat_dim = 8
        node_feat_type = 'node2vec'
        n2v_walk_length = 0.8
        n2v_num_walks = 100


class WSConfig(BaseConfig):
    class data:
        type = 'ws'
        load_data = False
        n_target_nodes = 64
        reduction_ratio = 0.5
        reduction_metric = 'degree'
        n_source_nodes = int(n_target_nodes * reduction_ratio)

        n_samples = 128

        k_min = 6
        k_max = 12
        p_min = 0.2
        p_max = 0.5

        node_feat_init = 'adj'
        node_feat_dim = 8
        node_feat_type = 'node2vec'
        n2v_walk_length = 0.8
        n2v_num_walks = 50