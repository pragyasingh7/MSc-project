import os
import datetime
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Logger:
    def __init__(self, config, exp_tag=''):
        self.exp_tag = exp_tag
        self.use_wandb = config.wandb.use_wandb
        self.res_dir = self._initialise_results_dir(config)

        if self.use_wandb:
            self._initialise_wandb_run(config)

    def _initialise_results_dir(self, config):
        """
        Initialise the results directory for the experiment.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        res_dir = os.path.join('results', config.dataset.dataset_name)
        res_dir = os.path.join(res_dir, f'epc{config.experiment.n_epochs}',
                                # f'bs{config.experiment.batch_size}', 
                                # f'lr{config.experiment.lr}',
                                f'{self.exp_tag}{str(timestamp)}')

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        return res_dir
    
    def _initialise_wandb_run(self, config):
        """
        Initialise the wandb run for the experiment.
        """
        wandb_config = {
            "exp_tag": self.exp_tag,
            "dataset": config.dataset.dataset_name,
            "model": config.model.model_name,
            "n_epochs": config.experiment.n_epochs,
            "batch_size": config.experiment.batch_size,
            "use_l1_loss": config.experiment.use_l1_loss,
            "use_topo_metric_loss": config.experiment.use_topo_metric_loss,
            "use_pearson_loss": config.experiment.use_pearson_loss,
        }

        if config.model.model_name == 'ggan':
            wandb_config = wandb_config | {
                "real_label": config.model.real_label,
                "fake_label": config.model.fake_label,
            }

        self.wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=wandb_config
        )

    def animate_results(self, 
                        eval_epochs, 
                        source_graphs, 
                        target_graphs, 
                        output_graphs,
                        sample_idx=0):
        """
        Show how evaluation results change over epochs.
        """
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        def update(i):
            ax[0].clear()
            ax[1].clear()
            ax[2].clear()

            source_graph = source_graphs[i][sample_idx]
            target_graph = target_graphs[i][sample_idx]
            output_graph = output_graphs[i][sample_idx]

            ax[0].imshow(source_graph, cmap='viridis')
            ax[0].set_title('Source Graph')

            ax[1].imshow(target_graph, cmap='viridis')
            ax[1].set_title('Target Graph')

            ax[2].imshow(output_graph, cmap='viridis')
            ax[2].set_title('Output Graph')

            fig.suptitle(f'Epoch {eval_epochs[i]}')

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=range(len(eval_epochs)), repeat=True)
        
        # Save animation
        ani_path = os.path.join(self.res_dir, 'evaluation.gif')
        ani.save(ani_path, writer='imagemagick', fps=1)

        if self.use_wandb:
            self.log_gif(ani_path, 'evaluation')

    def plot_results(self, source_graph, target_graph, output_graph):
        """
        Plot the source, target, and output graphs side by side.
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot source graph
        axs[0].imshow(source_graph, cmap='viridis')
        axs[0].set_title('Source Graph')

        # Plot target graph
        axs[1].imshow(target_graph, cmap='viridis')
        axs[1].set_title('Target Graph')

        # Plot output graph
        axs[2].imshow(output_graph, cmap='viridis')
        axs[2].set_title('Output Graph')

        # Save plot
        plot_path = os.path.join(self.res_dir, 'final_result.png')
        plt.savefig(plot_path)

        if self.use_wandb:
            self.log_image(plot_path, 'Final result')

    def log_image(self, image, tag):
        """
        Log an image to wandb.
        """
        if self.use_wandb:
            self.wandb_run.log({tag: wandb.Image(image)})

    def log_gif(self, gif_path, tag):
        """
        Log a gif to wandb.
        """
        if self.use_wandb:
            self.wandb_run.log({tag: wandb.Video(gif_path, format='gif')})

    def log_metric(self, metric, step, tag):
        """
        Log a metric to wandb.
        """
        if self.use_wandb:
            self.wandb_run.log({tag: metric}, step=step)

    def save_model(self, model, tag=''):
        """ 
        Save the PyTorch model's state dict.
        """
        model_path = os.path.join(self.res_dir, f'model{tag}.pt')
        torch.save(model.state_dict(), model_path)

        if self.use_wandb:
            self.wandb_run.save(model_path)

    def save_numpy_array(self, array, tag):
        """
        Save a numpy array to the results directory.
        """
        array_path = os.path.join(self.res_dir, f'{tag}.npy')
        np.save(array_path, array)

        if self.use_wandb:
            self.wandb_run.save(array_path)
    
    def finish(self):
        """
        Close the wandb run.
        """
        if self.use_wandb:
            self.wandb_run.finish()
