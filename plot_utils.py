import os
import tempfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_grad_flow(named_parameters, step, tmp_dir):
    """Save gradient flow plots in the temporary directory"""
    ave_grads = []
    max_grads = []
    layers = []
    # print('check grad nad all')
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())
    
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.plot(max_grads, alpha=0.3, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(f"Gradient flow, step {step}")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot as an image file in the temporary directory
    filename = os.path.join(tmp_dir, f"grad_flow_{step:03d}.png")
    plt.savefig(filename)
    plt.close()


def create_gif_grad(image_folder, gif_name):
    """Combine gradient flow plots into a GIF"""
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.startswith('grad_flow_') and file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))
    # print(len(os.listdir(image_folder)))
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=500, loop=0)


def format_colorbar(cb):
    """Function to format the colorbar"""
    cb.formatter = ScalarFormatter(useMathText=True)
    cb.formatter.set_scientific(False)
    cb.formatter.set_useOffset(False)
    cb.update_ticks()


def plot_adj_matrices(orig_s, orig_t, pred_t, step, tmp_dir, pred_s=None):
    """Plot the adjacency matrices of the source, target, and predicted graphs"""
    if pred_s is not None:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    cb = axs[0].imshow(orig_s, cmap='viridis')
    axs[0].set_title('Original Source')
    cb = plt.colorbar(cb, ax=axs[0])
    format_colorbar(cb)

    cb = axs[1].imshow(orig_t, cmap='viridis')
    axs[1].set_title('Original Target')
    cb = plt.colorbar(cb, ax=axs[1])
    format_colorbar(cb)

    cb = axs[2].imshow(pred_t, cmap='viridis')
    axs[2].set_title('Predicted Target')
    cb = plt.colorbar(cb, ax=axs[2])
    format_colorbar(cb)

    if pred_s is not None:
        cb = axs[3].imshow(pred_s, cmap='viridis')
        axs[3].set_title('Predicted Source')
        cb = plt.colorbar(cb, ax=axs[3])
        format_colorbar(cb)
    
    plt.tight_layout()
    
    # Save the plot as an image file in the temporary directory
    filename = os.path.join(tmp_dir, f"train_samples_{step:03d}.png")
    plt.savefig(filename)
    plt.close()


def create_gif_samples(image_folder, gif_name):
    """Combine adjacency matrix plots into a GIF"""
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.startswith('train_samples_') and file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=500, loop=0)


def plot_evaluated_sample(source_mat, target_mat, eval_mat, res_dir, eval_idx=6, eval_mat_s=None):
    # Plot source, target, and output

    if eval_mat_s is not None:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    cb = axs[0].imshow(source_mat)
    axs[0].set_title("Orig Source")
    cb = axs[0].figure.colorbar(cb, ax=axs[0])
    format_colorbar(cb)

    cb = axs[1].imshow(target_mat)
    axs[1].set_title("Orig Target")
    cb = axs[1].figure.colorbar(cb, ax=axs[1])
    format_colorbar(cb)

    cb = axs[2].imshow(eval_mat)
    axs[2].set_title("Pred Target")
    cb = axs[2].figure.colorbar(cb, ax=axs[2])
    format_colorbar(cb)

    if eval_mat_s is not None:
        cb = axs[3].imshow(eval_mat_s)
        axs[3].set_title("Pred Source")
        cb = axs[3].figure.colorbar(cb, ax=axs[3])
        format_colorbar(cb)

    plt.tight_layout()
    plt.savefig(f'{res_dir}eval_sample_{eval_idx}.png')
    plt.close()


def plot_average_predictions(source_mat_test, target_mat_test, eval_output_t, res_dir, eval_output_s=None):
    source_mat_test = [s.cpu().numpy() for s in source_mat_test]
    target_mat_test = [t.cpu().numpy() for t in target_mat_test]
    
    sample_list = [
        source_mat_test,
        target_mat_test,
        eval_output_t,
    ]
    titles = ['Source', 'Target', 'Target Output']

    if eval_output_s is not None:
        sample_list.append(eval_output_s)
        titles.append('Source Output')
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sample_mean = [np.mean(x, axis=0) for x in sample_list]

    for i, (sample, ax, title) in enumerate(zip(sample_list, axs, titles)):
        cb = ax.imshow(sample_mean[i])
        ax.set_title(title)
        cb = ax.figure.colorbar(cb, ax=ax)
        format_colorbar(cb)

    plt.tight_layout()
    plt.savefig(f'{res_dir}eval_mean.png')
    plt.close()


def plot_train_val_losses(train_losses, val_losses, res_dir):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(train_losses, label='Train Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train Loss')  

    axs[1].plot(val_losses, label='Val Loss', color='r')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Val Loss')

    plt.tight_layout()
    plt.savefig(f'{res_dir}train_val_losses.png')
    plt.close()


def plot_distribution_phy(plot_data, file_name='distribution'):
    mean_true = plot_data['mean_true']
    mean_pred = plot_data['mean_pred']
    mean_mae = plot_data['mean_mae']
    mean_mse = plot_data['mean_mse']

    std_true = plot_data['std_true']
    std_pred = plot_data['std_pred']
    std_mae = plot_data['std_mae']
    std_mse = plot_data['std_mse']

    # Plot global min and max for mean and std across true, pred, and mae
    mean_global_max = max(mean_true.max(), mean_pred.max(), mean_mae.max())
    mean_global_min = min(mean_true.min(), mean_pred.min(), mean_mae.min())

    std_global_max = max(std_true.max(), std_pred.max(), std_mae.max())
    std_global_min = min(std_true.min(), std_pred.min(), std_mae.min())

    # Plot the distribution as images
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Plot means on first row
    im = ax[0, 0].imshow(mean_true, cmap='viridis', vmin=mean_global_min, vmax=mean_global_max)
    ax[0, 0].set_title("Mean True Edge Features")

    ax[0, 1].imshow(mean_pred, cmap='viridis', vmin=mean_global_min, vmax=mean_global_max)
    ax[0, 1].set_title("Mean Predicted Edge Features")

    ax[0, 2].imshow(mean_mae, cmap='viridis', vmin=mean_global_min, vmax=mean_global_max)
    ax[0, 2].set_title("Mean Absolute Error")

    # Plot std on second row
    ax[1, 0].imshow(std_true, cmap='viridis', vmin=std_global_min, vmax=std_global_max)
    ax[1, 0].set_title("Std True Edge Features")

    ax[1, 1].imshow(std_pred, cmap='viridis', vmin=std_global_min, vmax=std_global_max)
    ax[1, 1].set_title("Std Predicted Edge Features")

    ax[1, 2].imshow(std_mae, cmap='viridis', vmin=std_global_min, vmax=std_global_max)
    ax[1, 2].set_title("Std Absolute Error")

    cbar_ax = fig.add_axes([0.1, 0.93, 0.8, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', fraction=0.02, pad=0.04)
    
    # Save the plot
    plt.savefig(f"{file_name}_mae.png")

    # plt.show()
    plt.close()

    # Plot MSE distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Get min-max for mse
    mse_global_max = max(mean_mse.max(), std_mse.max())
    mse_global_min = min(mean_mse.min(), std_mse.min())

    # Plot means on first row
    im = ax[0].imshow(mean_mse, cmap='viridis', vmin=mse_global_min, vmax=mse_global_max)
    ax[0].set_title("Mean of MSE")

    im = ax[1].imshow(std_mse, cmap='viridis', vmin=mse_global_min, vmax=mse_global_max)
    ax[1].set_title("Std of MSE")

    cbar_ax = fig.add_axes([0.1, 0.93, 0.8, 0.03])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', fraction=0.02, pad=0.04)

    # Save the plot
    plt.savefig(f"{file_name}_mse.png")
    
    # plt.show()
    plt.close()