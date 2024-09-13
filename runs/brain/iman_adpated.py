import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
from tqdm import tqdm
from sklearn.model_selection import KFold

from models.adapted_imangraphnet import load_iman_model
from configs.brain import Config
from data.connectomes import load_dataset
from runs.utils import set_seed
from plot_utils import plot_adj_matrices, plot_train_val_losses, create_gif_samples, plot_evaluated_sample, plot_average_predictions


# Define alignment loss
def alignment_loss(target, predicted):
    kl_loss = torch.abs(F.kl_div(F.softmax(target, dim=-1), F.softmax(predicted, dim=-1), None, None, 'sum'))
    kl_loss = (1/350) * kl_loss
    return kl_loss


set_seed()

# Load configs
config = Config()

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

run_dir = f"{config.experiment.base_dir}/adapted_iman/{config.experiment.run_name}/"     

for fold, (train_idx, test_idx) in enumerate(kf.split(source_pyg_all)):
    print(f"Fold {fold+1}/3")

    # Cretae fold directory
    res_dir = f'{run_dir}fold_{fold+1}/'
    os.makedirs(res_dir, exist_ok=True)

    # Initialize model, optimizer, and loss function
    model = load_iman_model(config).to(device)
    aligner_optimizer = torch.optim.Adam(model.aligner.parameters(), lr=0.025, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=0.025, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))

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
            epoch_al_loss = 0.0
            epoch_g_loss = 0.0
            epoch_gg_loss = 0.0
            epoch_d_loss = 0.0

            batch_counter = 0

            model.train()

            # Shuffle training data
            random_idx = torch.randperm(len(source_pyg_train))
            source_pyg = [source_pyg_train[i] for i in random_idx]
            target_pyg = [target_pyg_train[i] for i in random_idx]
            source_mat = [source_mat_train[i] for i in random_idx]
            target_mat = [target_mat_train[i] for i in random_idx]

            for source_g, source_m, target_g, target_m in tqdm(zip(source_pyg, source_mat, target_pyg, target_mat), total=len(source_pyg)):
                pred, al_loss, g_loss, gg_loss, d_loss = model(source_g, target_g)

                al_loss.backward(retain_graph=True)
                g_loss.backward(retain_graph=True)
                d_loss.backward()

                epoch_al_loss += al_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_gg_loss += gg_loss.item()
                epoch_d_loss += d_loss.item()

                batch_counter += 1

                # Do mini-batch gradient descent
                if batch_counter % config.experiment.batch_size == 0 or batch_counter == len(source_pyg):
                    aligner_optimizer.step()
                    generator_optimizer.step()
                    discriminator_optimizer.step()

                    aligner_optimizer.zero_grad()
                    generator_optimizer.zero_grad()
                    discriminator_optimizer.zero_grad()

                    step += 1

                    # Plot adj matrices
                    source_t = source_m.detach().cpu().numpy()

                    pred = pred.detach().cpu().numpy()
                    target = target_m.detach().cpu().numpy()

                    plot_adj_matrices(source_t, target, pred, step, tmp_dir)

                    torch.cuda.empty_cache()
                    gc.collect()

            epoch_al_loss = epoch_al_loss / len(source_pyg_train)
            epoch_g_loss = epoch_g_loss / len(source_pyg_train)
            epoch_gg_loss = epoch_gg_loss / len(source_pyg_train)
            epoch_d_loss = epoch_d_loss / len(source_pyg_train)

            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Aligner Loss: {epoch_al_loss}")
            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Generator Loss: {epoch_g_loss}")
            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Generator Loss (L1): {epoch_gg_loss}")
            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Discriminator Loss: {epoch_d_loss}")
            train_losses.append(epoch_gg_loss)

            # Validation 
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for source_g, target_g in zip(source_pyg_val, target_pyg_val):
                    _, _, _, gg_loss, _ = model(source_g, target_g)

                    val_loss += gg_loss.item()

                    torch.cuda.empty_cache()
                    gc.collect()

            val_loss = val_loss / len(source_pyg_val)

            print(f"Epoch {epoch+1}/{config.experiment.max_epochs}, Val gen Loss: {val_loss}")
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
        eval_loss_t = []

        with torch.no_grad():
            for source_g, target_g in zip(source_pyg_test, target_pyg_test):
                g_output, _, _, gg_loss, _ = model(source_g, target_g)

                eval_output_t.append(g_output.detach().cpu().numpy())
                eval_loss_t.append(gg_loss.item())

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