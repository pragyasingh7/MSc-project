class Config:
    class experiment:
        # Parameters for K-fold cross validation
        num_splits = 3
        shuffle = True
        random_state = 42

        # Training parameters
        max_epochs = 300
        lr = 0.001
        batch_size = 16

        # Early-stopping parameters
        early_stopping = True
        patience = 7
        warm_up_epochs = 30

        # Logging results
        base_dir = 'results'
        run_name = 'brain'

    class model:
        sr_method = 'linear_algebraic'  # 'linear_algebraic', 'bi_lc', 'bi_mp'
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


    # Parameters for the connectomics dataset
    class data:
        type = 'connectomics'
        
        n_source_nodes = 160
        n_target_nodes = 268

        source_dir = "data/lr_train.csv"
        target_dir = "data/hr_train.csv"

        node_feat_init = 'adj'
        node_feat_dim = 1
