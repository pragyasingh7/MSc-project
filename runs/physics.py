import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data.phy import create_grid_graphs, create_random_geometric_graphs, convert_edge_feat_array_to_matrix
from models.phy import NodeModel, NodeLargeModel, EdgeModel, DualEdgeModel
from configs.physics import Config
from plot_utils import plot_distribution_phy


def evaluate_test_graphs(model, graphs, file_name='distribution.png', edge_model=False):
    # Log the distribution of edge features
    true_edge_feats = []
    pred_edge_feats = []
    mae_edge_feats = []
    mse_edge_feats = []

    test_mae = 0.0
    test_mse = 0.0

    for node_feats, edge_idx, edge_feats, A, num_nodes in graphs:
        edge_feats_pred = model(node_feats, edge_idx)

        if edge_model:
            edge_feats_pred = convert_edge_feat_array_to_matrix(edge_feats_pred, edge_idx.T, num_nodes)
        
        edge_mask = torch.tensor(A.toarray()).to(edge_feats_pred.device)
        edge_feats_pred = edge_feats_pred * edge_mask

        edge_feats_target = convert_edge_feat_array_to_matrix(edge_feats, edge_idx.T, num_nodes)

        true_edge_feats.append(edge_feats_target.cpu().detach().numpy())
        pred_edge_feats.append(edge_feats_pred.cpu().detach().numpy())    
        mae_edge_feats.append(abs(edge_feats_target - edge_feats_pred).cpu().detach().numpy())
        mse_edge_feats.append((edge_feats_target - edge_feats_pred).detach().cpu().numpy()**2)

        test_mae += abs(edge_feats_target - edge_feats_pred).sum().item() / edge_idx.shape[1]
        test_mse += ((edge_feats_target - edge_feats_pred)**2).sum().item() / edge_idx.shape[1]

    test_mae /= len(graphs)
    test_mse /= len(graphs)

    # Calculate mean across all val graphs using numpy
    mean_true = np.mean(true_edge_feats, axis=0)
    mean_pred = np.mean(pred_edge_feats, axis=0)
    mean_mae = np.mean(mae_edge_feats, axis=0)
    mean_mse = np.mean(mse_edge_feats, axis=0)
    

    # Calculate std across all val graphs
    std_true = np.std(true_edge_feats, axis=0)
    std_pred = np.std(pred_edge_feats, axis=0)
    std_mae = np.std(mae_edge_feats, axis=0)
    std_mse = np.std(mse_edge_feats, axis=0)


    plot_data = {
        'mean_true': mean_true,
        'mean_pred': mean_pred,
        'mean_mae': mean_mae,
        'mean_mse': mean_mse,

        'std_true': std_true,
        'std_pred': std_pred,
        'std_mae': std_mae,
        'std_mse': std_mse,
    }

    plot_distribution_phy(plot_data, file_name)

    return test_mae, test_mse, plot_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()

max_epochs = config.max_epochs
batch_size = config.batch_size
lr = config.lr

early_stop = config.early_stop
warm_up_preiod = config.warm_up_preiod
patience = config.patience

run_name = config.run_name
base_dir = config.base_dir

res_dir = f"{base_dir}/{run_name}/"
os.makedirs(res_dir, exist_ok=True)

# Dataset parameters
graph_type = config.graph_type
eq_type = config.eq_type
uniform_mass = config.uniform_mass
data_gen_seed = config.data_gen_seed

n_nodes = config.n_nodes
threshold = config.threshold

grid_size = config.grid_size

n_train_samples = config.n_train_samples
n_val_samples = config.n_val_samples
n_test_samples = config.n_test_samples

# Model type parameters
edge_model = config.edge_model
dual_edge_model = config.dual_edge_model
node_large = config.node_large


results = {
    "test_mae": 0.0,
    "test_mse": 0.0,
    "plot_data": {},
}

# Early stopping variables
es_counter = 0
best_val_loss = np.inf
best_val_model = None   # To store state_dict of the best model

# Parameter check
if max_epochs < warm_up_preiod:
    raise ValueError("Warm up period should be less than max epochs")

# Load model
if edge_model:
    if dual_edge_model:
        model = DualEdgeModel().to(device)
    else:
        model = EdgeModel().to(device)
elif node_large:
    model = NodeLargeModel().to(device)
else:
    model = NodeModel().to(device)

n_total_samples = n_train_samples + n_val_samples + n_test_samples
# Load dataset
if graph_type == "grid":
    all_graphs = create_grid_graphs(n_total_samples, grid_size, uniform_mass=uniform_mass, 
                                    eq_type=eq_type, device=device)

elif graph_type == "random_geometric":
    all_graphs = create_random_geometric_graphs(n_total_samples, n_nodes, threshold, 
                                                uniform_mass=uniform_mass, data_gen_seed=data_gen_seed,
                                                eq_type=eq_type, device=device)

# Split in train, val, and test
train_graphs = all_graphs[:n_train_samples]
val_graphs = all_graphs[n_train_samples:n_train_samples+n_val_samples]
test_graphs = all_graphs[n_train_samples+n_val_samples:]

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='sum')   

train_losses = []
val_losses = []

for epoch in range(max_epochs):
    batch_counter = 0

    model.train()
    epoch_train_loss = 0

    # Shuffle the training graphs out of place
    shuffle_idx = np.arange(len(train_graphs))
    train_graphs_shuffled = [train_graphs[i] for i in shuffle_idx]

    for node_feats, edge_idx, edge_feats, A, num_nodes in train_graphs_shuffled:        
        edge_feats_pred = model(node_feats, edge_idx)

        if edge_model:
            edge_feats_target = edge_feats
        else:
            edge_mask = torch.tensor(A.toarray()).to(device)
            edge_feats_pred = edge_feats_pred * edge_mask

            edge_feats_target = convert_edge_feat_array_to_matrix(edge_feats, edge_idx.T, num_nodes)

        loss = loss_fn(edge_feats_pred, edge_feats_target) / edge_idx.shape[1]
        epoch_train_loss += loss.item()

        loss.backward()

        batch_counter += 1

        if batch_counter % batch_size == 0 or batch_counter == len(train_graphs):
            optimizer.step()
            optimizer.zero_grad()

    epoch_train_loss /= len(train_graphs)
    print(f"Epoch {epoch} - Training Loss: {epoch_train_loss}")
    train_losses.append(epoch_train_loss)

    
    # Get epoch validation loss
    model.eval()
    epoch_val_loss = 0

    with torch.no_grad():
        for node_feats, edge_idx, edge_feats, A, num_nodes in val_graphs:
            edge_feats_pred = model(node_feats, edge_idx)

            if edge_model:
                edge_feats_target = edge_feats
            else:
                edge_mask = torch.tensor(A.toarray()).to(device)
                edge_feats_pred = edge_feats_pred * edge_mask

                edge_feats_target = convert_edge_feat_array_to_matrix(edge_feats, edge_idx.T, num_nodes)

            loss = loss_fn(edge_feats_pred, edge_feats_target) / edge_idx.shape[1]
            epoch_val_loss += loss.item()

    epoch_val_loss /= len(val_graphs)
    print(f"Epoch {epoch} - Validation Loss: {epoch_val_loss}")
    val_losses.append(epoch_val_loss)

    if epoch > warm_up_preiod:
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_model = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1

        if epoch == max_epochs - 1:
            print("Max epochs reached")
            test_mae, test_mse, plot_data = evaluate_test_graphs(model, test_graphs, 
                                                file_name=f"{res_dir}distribution", edge_model=edge_model)

        if early_stop and es_counter >= patience:
            print("Early stopping")

            # Log the distribution of edge features
            model.load_state_dict(best_val_model)   # Load the best model
            test_mae, test_mse, plot_data = evaluate_test_graphs(model, test_graphs, 
                                                file_name=f"{res_dir}distribution.png", edge_model=edge_model)
            
            break

# Print best validation loss
print(f"Best Validation Loss: {best_val_loss}")

# Performance on the test set
print(f"Test MAE: {test_mae}")
print(f"Test MSE: {test_mse}")

results["test_mae"] = test_mae
results["test_mse"] = test_mse
results["plot_data"] = plot_data