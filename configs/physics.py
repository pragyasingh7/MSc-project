class Config:
    class experiment:
        run_name = "physics"
        n_runs = 15

    class model:
        node_large = False
        edge_model = False
        dual_edge_model = False

        max_epochs = 300
        batch_size = 16
        lr = 0.001

        early_stop = True
        warm_up_preiod = 10
        patience = 15

    class data:
        eq_type = 'e1'                      # 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'
        graph_type = "random_geometric"     # 'random_geometric', 'grid'
        uniform_mass = True
        data_gen_seed = 99
        base_dir = f"results/{graph_type}/{eq_type}/"

        # For random geometric graphs
        n_nodes = 16 
        threshold = 0.3

        # For grid graphs
        grid_size = 4

        n_train_samples = 128
        n_val_samples = 32
        n_test_samples = 32
        n_total_samples = n_train_samples + n_val_samples + n_test_samples