import os
import pickle
import gc
import torch
import numpy as np
import torch.nn as nn
import tempfile
from tqdm import tqdm
from sklearn.model_selection import KFold

from models.dual_sr import load_dual_model
from models.bi_sr import load_bi_sr_model
from configs.simulated import DATASET_TYPE, SBMConfig, BAConfig, WSConfig
from data.utils import create_dual_graph, create_dual_graph_feature_matrix, revert_dual
from runs.utils import set_seed
from plot_utils import plot_adj_matrices, plot_train_val_losses, create_gif_samples, plot_evaluated_sample, plot_average_predictions


def load_model(config, device):
    target_node_embeddings = None

    if config.model.sr_method == 'bi_mp':
        # Fixed target node embedding to specify target node order
        target_node_embeddings = torch.randn(config.data.n_target_nodes, 
                                            config.model.hidden_dim * config.model.num_heads, 
                                            device=device)
    if config.model.use_dual:
        model = load_dual_model(config, target_node_embeddings)
    else:
        model = load_bi_sr_model(config, target_node_embeddings)

    return model

def load_configs():
    if DATASET_TYPE == 'sbm':
        config = SBMConfig()
    elif DATASET_TYPE == 'ba':
        config = BAConfig()
    elif DATASET_TYPE == 'ws':
        config = WSConfig()
    else:
        raise ValueError(f"Invalid dataset type: {DATASET_TYPE}")
    
    return config

def load_dataset(config, device):
    if config.data.type == 'sbm':
        from data.sbm import load_dataset
    elif config.data.type == 'ba':
        from data.ba import load_dataset
    elif config.data.type == 'ws':
        from data.ws import load_dataset
    else:
        raise ValueError(f"Invalid dataset type: {config.data.type}")

    return load_dataset(config, device)

set_seed()

config = load_configs()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
all_dataset = load_dataset(config, device)
source_pyg_all, target_pyg_all, source_mat_all, target_mat_all = all_dataset

print("Finished loading dataset")

# K-fold cross validation
kf = KFold(n_splits=config.experiment.num_splits, 
           shuffle=config.experiment.shuffle, 
           random_state=config.experiment.random_state)

run_dir = f"{config.experiment.base_dir}{config.experiment.run_name}/"
run_dir = f"{run_dir}{'dual' if config.model.use_dual else 'directSR'}/"
run_dir = f"{run_dir}{config.model.sr_method}{'_refine' if config.model.refine_target else '/'}"
if config.model.refine_target:
    run_dir = f"{run_dir}{'_fixed' if config.model.target_refine_fully_connected else '_learn'}/"

print(f"Run dir: {run_dir}")

# Initialize dual domain graph
if config.model.use_dual:
    n_target_nodes = config.data.n_target_nodes
    dual_domain = torch.ones((n_target_nodes, n_target_nodes), dtype=torch.float, device=device)
    dual_edge_index, _ = create_dual_graph(dual_domain)

for fold, (train_idx, test_idx) in enumerate(kf.split(source_pyg_all)):
    print(f"Fold {fold+1}/3")

    res_dir = f'{run_dir}fold_{fold+1}/'
    os.makedirs(res_dir, exist_ok=True)

    # Initialize model, optimizer, and loss function
    model = load_model(config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.experiment.lr)
    critereon = nn.MSELoss()

    # Split train idx further into train and val idx
    train_size = int(0.8 * len(train_idx))
    train_idx, val_idx = train_idx[:train_size], train_idx[train_size:]

    # Split into training, val, and test sets
    source_pyg_train = [source_pyg_all[i] for i in train_idx]
    target_pyg_train = [target_pyg_all[i] for i in train_idx]
    source_mat_train = [source_mat_all[i] for i in train_idx]
    target_mat_train = [target_mat_all[i] for i in train_idx]

    source_pyg_val = [source_pyg_all[i] for i in val_idx]
    target_pyg_val = [target_pyg_all[i] for i in val_idx]
    source_mat_val = [source_mat_all[i] for i in val_idx]
    target_mat_val = [target_mat_all[i] for i in val_idx]

    source_pyg_test = [source_pyg_all[i] for i in test_idx]
    target_pyg_test = [target_pyg_all[i] for i in test_idx]
    source_mat_test = [source_mat_all[i] for i in test_idx]
    target_mat_test = [target_mat_all[i] for i in test_idx]

    # Log epoch losses
    train_losses = []
    val_losses = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        step = 0
        es_counter = 0
        best_val_loss = float('inf')
        best_val_model = None

        for epoch in range(config.experiment.max_epochs):
            epoch_loss = 0.0
            batch_counter = 0

            model.train()

            # Shuffle training data
            random_idx = torch.randperm(len(source_pyg_train))
            source_pyg = [source_pyg_train[i] for i in random_idx]
            source_mat = [source_mat_train[i] for i in random_idx]
            target_mat = [target_mat_train[i] for i in random_idx]

            for source_g, source_m, target_m in tqdm(zip(source_pyg, source_mat, target_mat), total=len(source_pyg)):
                if config.model.use_dual:
                    pred = model(source_g, dual_edge_index)
                    target = create_dual_graph_feature_matrix(target_m)
                else:
                    pred = model(source_g)
                    target = target_m

                loss = critereon(pred, target)
                loss.backward()

                epoch_loss += loss.item()
                batch_counter += 1

                # Do mini-batch gradient descent
                if batch_counter % config.experiment.batch_size == 0 or batch_counter == len(source_pyg):
                    optimizer.step()
                    optimizer.zero_grad()

                    step += 1

                    # Plot adj matrices
                    source_t = source_m.detach().cpu().numpy()

                    pred = pred.detach()
                    target = target.detach()

                    if config.model.use_dual:
                        pred = revert_dual(pred.detach(), n_target_nodes)
                        target = revert_dual(target.detach(), n_target_nodes)

                    pred = pred.cpu().numpy()
                    target = target.cpu().numpy()

                    plot_adj_matrices(source_t, target, pred, step, tmp_dir)

                    torch.cuda.empty_cache()
                    gc.collect()

            epoch_loss = epoch_loss / len(source_pyg_train)
            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Train Loss: {epoch_loss}")
            train_losses.append(epoch_loss)

            # Validation 
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for source_g, source_m, target_m in zip(source_pyg_val, source_mat_val, target_mat_val):
                    if config.model.use_dual:
                        pred = model(source_g, dual_edge_index)
                        target = create_dual_graph_feature_matrix(target_m)
                    else:
                        pred = model(source_g)
                        target = target_m

                    loss = critereon(pred, target)
                    val_loss += loss.item()

            torch.cuda.empty_cache()
            gc.collect()

            val_loss = val_loss / len(source_pyg_val)

            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Val Loss: {val_loss}")
            val_losses.append(val_loss)

            # Check for early-stopping
            if config.experiment.early_stopping:
                if epoch < config.experiment.warm_up_epochs:
                    continue

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_model = model.state_dict()
                    es_counter = 0

                else:
                    es_counter += 1

                if es_counter >= config.experiment.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    model.load_state_dict(best_val_model)
                    break

        # Save train and val losses
        np.save(f'{res_dir}train_losses.npy', np.array(train_losses))
        np.save(f'{res_dir}val_losses.npy', np.array(val_losses))

        # Discarding the loss from first epoch since it's usually very high
        plot_train_val_losses(train_losses[1:], val_losses[1:], res_dir)

        # Evaluate on test set
        eval_output_t = []
        eval_loss_t = []

        with torch.no_grad():
            for source_g, source_m, target_m in zip(source_pyg_test, source_mat_test, target_mat_test):
                if config.model.use_dual:
                    pred = model(source_g, dual_edge_index)
                    target = create_dual_graph_feature_matrix(target_m)
                    eval_loss_t.append(critereon(pred, target).item())
                    pred = revert_dual(pred.detach(), n_target_nodes)
                else:
                    pred = model(source_g)
                    eval_loss_t.append(critereon(pred, target_m).item())

                eval_output_t.append(pred.cpu().numpy())

        eval_loss_t = np.mean(eval_loss_t)
        print(f"Test Loss (Target): {eval_loss_t}")

        # Save eval_output_t
        np.save(f'{res_dir}eval_output.npy', np.array(eval_output_t))

        # Create gif of training samples
        gif_path = f"{res_dir}train_samples.gif"
        create_gif_samples(tmp_dir, gif_path)
        print(f"Training samples saved as {gif_path}")

        # Save model
        model_path = f"{res_dir}model.pth"
        torch.save(model.state_dict(), model_path)

        # Plot eval results
        eval_idx = 0
        source_mat = source_mat_test[eval_idx].cpu().numpy()
        target_mat = target_mat_test[eval_idx].cpu().numpy()
        eval_mat = eval_output_t[eval_idx]
        plot_evaluated_sample(source_mat, target_mat, eval_mat, res_dir, eval_idx=eval_idx)

        plot_average_predictions(source_mat_test, target_mat_test, eval_output_t, res_dir)