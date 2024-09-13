import os
import gc
import torch
import numpy as np
import torch.nn as nn
import tempfile
from tqdm import tqdm
from sklearn.model_selection import KFold

from models.autoencoder import load_ae_model
from configs.brain import Config
from data.connectomes import load_dataset
from runs.utils import set_seed
from plot_utils import plot_adj_matrices, plot_train_val_losses, create_gif_samples, plot_evaluated_sample, plot_average_predictions


set_seed()

# Load configs
config = Config()
n_source_nodes = config.data.n_source_nodes
n_target_nodes = config.data.n_target_nodes

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
all_dataset = load_dataset(config, device)
source_pyg_all, target_pyg_all, source_mat_all, target_mat_all = all_dataset

print('Finished loading dataset')

# K-fold cross validation
kf = KFold(n_splits=config.experiment.num_splits, 
           shuffle=config.experiment.shuffle, 
           random_state=config.experiment.random_state)

print('created splits')

run_dir = f"{config.experiment.base_dir}/autoencoder/{config.experiment.run_name}/"     

for fold, (train_idx, test_idx) in enumerate(kf.split(source_pyg_all)):
    print(f"Fold {fold+1}/3")

    # Cretae fold directory
    res_dir = f'{run_dir}fold_{fold+1}/'
    os.makedirs(res_dir, exist_ok=True)

    # Initialize model, optimizer, and loss function
    model = load_ae_model(config, n_source_nodes, n_target_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.experiment.lr)
    criterion = nn.L1Loss()

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
            epoch_lr_loss = 0.0
            epoch_hr_loss = 0.0
            batch_counter = 0

            model.train()

            # Shuffle training data
            random_idx = torch.randperm(len(source_pyg_train))
            source_pyg = [source_pyg_train[i] for i in random_idx]
            target_pyg = [target_pyg_train[i] for i in random_idx]
            source_mat = [source_mat_train[i] for i in random_idx]
            target_mat = [target_mat_train[i] for i in random_idx]

            for source_g, source_m, target_m in tqdm(zip(source_pyg, source_mat, target_mat), total=len(source_pyg)):
                lr_pred, hr_pred = model(source_g)

                lr_loss = criterion(lr_pred, source_m)
                hr_loss = criterion(hr_pred, target_m)

                hr_loss.backward(retain_graph=True)
                lr_loss.backward()

                epoch_lr_loss += lr_loss.item()
                epoch_hr_loss += hr_loss.item()

                batch_counter += 1

                # Do mini-batch gradient descent
                if batch_counter % config.experiment.batch_size == 0 or batch_counter == len(source_pyg):
                    optimizer.step()
                    optimizer.zero_grad()

                    step += 1

                    # Plot adj matrices
                    source_t = source_m.detach().cpu().numpy()

                    hr_pred = hr_pred.detach().cpu().numpy()
                    lr_pred = lr_pred.detach().cpu().numpy()
                    target = target_m.detach().cpu().numpy()

                    plot_adj_matrices(source_t, target, hr_pred, step, tmp_dir, pred_s=lr_pred)

                    torch.cuda.empty_cache()
                    gc.collect()

            epoch_lr_loss = epoch_lr_loss / len(source_pyg)
            epoch_hr_loss = epoch_hr_loss / len(source_pyg)

            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, LR Loss: {epoch_lr_loss}")
            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, HR Loss: {epoch_hr_loss}")

            train_losses.append(epoch_hr_loss)

            # Validation 
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for source_g, target_m in zip(source_pyg_val, target_mat_val):
                    lr_pred, hr_pred = model(source_g)

                    hr_loss = criterion(hr_pred, target_m)

                    val_loss += hr_loss.item()

                    torch.cuda.empty_cache()
                    gc.collect()

            val_loss = val_loss / len(source_pyg_val)

            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Val HR Loss: {val_loss}")
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

        plot_train_val_losses(train_losses, val_losses, res_dir)

        # Evaluate on test set
        eval_output_t = []
        eval_output_s = []
        eval_loss_t = []
        eval_loss_s = []

        with torch.no_grad():
            for source_g, target_m in zip(source_pyg_test, target_mat_test):
                lr_pred, hr_pred = model(source_g)
                hr_loss = criterion(hr_pred, target_m)
                lr_loss = criterion(lr_pred, source_m)
                eval_output_t.append(hr_pred.detach().cpu().numpy())
                eval_output_s.append(lr_pred.detach().cpu().numpy())
                eval_loss_t.append(hr_loss.item())
                eval_loss_s.append(lr_loss.item())

        eval_loss_t = np.mean(eval_loss_t)
        print(f"Test Loss (Target): {eval_loss_t}")

        eval_loss_s = np.mean(eval_loss_s)
        print(f"Test Loss (Source): {eval_loss_s}")

        # Save eval_output_t
        np.save(f'{res_dir}eval_output.npy', np.array(eval_output_t))
        np.save(f'{res_dir}eval_output_s.npy', np.array(eval_output_s))

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
        eval_mat_s = eval_output_s[eval_idx]
        plot_evaluated_sample(source_mat, target_mat, eval_mat, res_dir, eval_idx=eval_idx, eval_mat_s=eval_mat_s)

        plot_average_predictions(source_mat_test, target_mat_test, eval_output_t, res_dir, eval_output_s=eval_output_s)