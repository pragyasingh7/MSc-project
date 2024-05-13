import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

from src.matrix_vectorizer import MatrixVectorizer
from src.models.ggan import GCNDiscriminator, GCNGenerator
from src.logging import Logger
from src.utils import create_pyg_graph
from src.losses import TopoMetricLoss, PearsonCorrelationLoss


class GANTrainer:
    def __init__(self, config, exp_tag=''):
        self.n_source_nodes = config.dataset.n_source_nodes
        self.n_target_nodes = config.dataset.n_target_nodes
        self.n_epochs = config.experiment.n_epochs

        # Specify additional generator losses
        self.use_l1_loss = config.experiment.use_l1_loss
        self.use_topo_metric_loss = config.experiment.use_topo_metric_loss
        self.use_pearson_loss = config.experiment.use_pearson_loss

        self.save_checkpoints = config.experiment.save_checkpoints
        self.checkpoint_freq = config.experiment.checkpoint_freq
        self.eval_freq = config.experiment.eval_freq
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matrix_vectorizer = MatrixVectorizer()
        self.logger = Logger(config, exp_tag)
        
        self.setup_trainer(config)


    def setup_trainer(self, config):
        """
        Set up different training compoenents.
        """
        # Define the generator model
        self.generator = GCNGenerator(
            self.n_source_nodes,
            self.n_target_nodes,
            cached=config.model.generator.cached,
            bn_eps=config.model.generator.bn_eps,
            bn_momentum=config.model.generator.bn_momentum,
            bn_affine=config.model.generator.bn_affine,
            bn_track_running_stats=config.model.generator.bn_track_running_stats
        )
        self.generator.to(self.device)


        # Define the discriminator model
        self.discriminator = GCNDiscriminator(
            self.n_source_nodes,
            self.n_target_nodes,
            cached=config.model.discriminator.cached,
            bn_eps=config.model.discriminator.bn_eps,
            bn_momentum=config.model.discriminator.bn_momentum,
            bn_affine=config.model.discriminator.bn_affine,
            bn_track_running_stats=config.model.discriminator.bn_track_running_stats
        )
        self.discriminator.to(self.device)


        # Define losses
        self.adversarial_loss = torch.nn.BCELoss()
        self.adversarial_loss.to(self.device)

        self.l1_loss = torch.nn.L1Loss()
        self.l1_loss.to(self.device)

        self.topo_loss = TopoMetricLoss()
        self.topo_loss.to(self.device)

        self.pearson_loss = PearsonCorrelationLoss()
        self.pearson_loss.to(self.device)

        # Define the optimisers
        self.generator_optimiser = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.model.generator.lr,
            betas=(config.model.generator.beta1, config.model.generator.beta2)
        )
        self.discriminator_optimiser = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.model.discriminator.lr,
            betas=(config.model.discriminator.beta1, config.model.discriminator.beta2)
        )   

    
    def prepare_data(self, source_data, target_data):
        """
        Pre-process the data and convert to PyG Data objects.
        """
        # TODO: Add batching
        # Convert vector to adjacency matrix
        source_data = [self.matrix_vectorizer.anti_vectorize(vector, self.n_source_nodes) 
                        for vector in source_data]
        target_data = [self.matrix_vectorizer.anti_vectorize(vector, self.n_target_nodes)
                        for vector in target_data]

        # Convert adjacency matrix to PyG graph
        source_data = [create_pyg_graph(matrix, self.n_source_nodes) for matrix in source_data]
        target_data = [create_pyg_graph(matrix, self.n_target_nodes) for matrix in target_data]
        
        return source_data, target_data
    

    def train_epoch(self, source_data, target_data, epoch):
        """
        Train the GAN model for a single epoch.
        """
        self.generator.train()
        self.discriminator.train()

        with torch.autograd.set_detect_anomaly(True):
            epoch_gen_loss = []
            epoch_dis_loss = []

            # Additional generator lossess
            epoch_gen_adv_loss = []
            epoch_gen_l1_loss = []
            epoch_gen_topo_loss = []
            epoch_gen_pearson_loss = []

            # Discriminator predictions
            real_dis_pred = []
            fake_dis_pred = []

            for source_graph, target_graph in tqdm(zip(source_data, target_data), desc=f"Training epoch {epoch}", total=len(source_data)):
                source_graph = source_graph.to(self.device)
                target_graph = target_graph.to(self.device)

                # Generator output
                gen_output = self.generator(source_graph)
                gen_graph = create_pyg_graph(gen_output, self.n_target_nodes)

                # print('Gen input shape: ', source_graph.x.shape)
                # print('Gen output shape: ', gen_output.shape)

                # Discriminator output
                dis_real = self.discriminator(target_graph)
                dis_fake = self.discriminator(gen_graph)

                # print('Dis real:', dis_real.shape)
                # print('Dis fake:', dis_fake.shape)

                real_dis_pred.append(dis_real.item())
                fake_dis_pred.append(dis_fake.item())

                # Generator loss
                gen_loss = self.adversarial_loss(dis_fake, torch.ones_like(dis_fake))
                epoch_gen_adv_loss.append(gen_loss.clone().detach())
                
                if self.use_l1_loss:
                    gen_l1 = self.l1_loss(gen_output, target_graph.x)
                    gen_loss += gen_l1
                    epoch_gen_l1_loss.append(gen_l1)

                if self.use_topo_metric_loss:
                    gen_topo_metric = self.topo_loss(gen_output, target_graph.x)
                    gen_loss += gen_topo_metric
                    epoch_gen_topo_loss.append(gen_topo_metric)

                if self.use_pearson_loss:
                    gen_pearson = self.pearson_loss(gen_output, target_graph.x)
                    gen_loss += gen_pearson
                    epoch_gen_pearson_loss.append(gen_pearson)

                # epoch_gen_loss.append(gen_loss.item())
                epoch_gen_loss.append(gen_loss)

                # Discriminator loss
                dis_real_loss = self.adversarial_loss(dis_real, torch.ones_like(dis_real))
                dis_fake_loss = self.adversarial_loss(dis_fake, torch.zeros_like(dis_fake))
                dis_loss = (dis_real_loss + dis_fake_loss) / 2
                # epoch_dis_loss.append(dis_loss.item())
                epoch_dis_loss.append(dis_loss)

            # Calculate losses for the complete epoch
            epoch_gen_loss = torch.stack(epoch_gen_loss).mean()
            epoch_gen_adv_loss = torch.stack(epoch_gen_adv_loss).mean()

            epoch_dis_loss = torch.stack(epoch_dis_loss).mean()
            
            # Log generator losses
            self.logger.log_metric(epoch_gen_loss.item(), epoch, 'Generator loss')
            self.logger.log_metric(epoch_gen_adv_loss.item(), epoch, 'Gen Adversarial loss')

            if self.use_l1_loss:
                epoch_gen_l1_loss = torch.stack(epoch_gen_l1_loss).mean()
                self.logger.log_metric(epoch_gen_l1_loss.item(), epoch, 'Gen L1 loss')

            if self.use_topo_metric_loss:
                epoch_gen_topo_loss = torch.stack(epoch_gen_topo_loss).mean()
                self.logger.log_metric(epoch_gen_topo_loss.item(), epoch, 'Gen Topological metric loss')
            
            if self.use_pearson_loss:
                epoch_gen_pearson_loss = torch.stack(epoch_gen_pearson_loss).mean()
                self.logger.log_metric(epoch_gen_pearson_loss.item(), epoch, 'Gen Pearson correlation loss')

            # Log discriminator predictions
            self.logger.log_metric(np.mean(real_dis_pred), epoch, 'Discriminator prediction on real data')
            self.logger.log_metric(np.mean(fake_dis_pred), epoch, 'Discriminator prediction on generated data')

            # Backpropagate generator loss
            self.generator_optimiser.zero_grad()                    
            epoch_gen_loss.backward(retain_graph=True)
            # epoch_gen_loss.backward()
            self.generator_optimiser.step()

            # Log discriminator loss
            self.logger.log_metric(epoch_dis_loss.item(), epoch, 'Discriminator loss') 
            
            # Backpropagate discriminator loss
            self.discriminator_optimiser.zero_grad()
            # epoch_dis_loss.backward(retain_graph=True)
            epoch_dis_loss.backward()
            self.discriminator_optimiser.step()

            # Report losses
            print(f"Train === Epoch: {epoch} | Generator loss: {epoch_gen_loss.item():.3f} | Discriminator loss: {epoch_dis_loss.item():.3f}")

            # Save checkpoints
            if self.save_checkpoints and epoch % self.checkpoint_freq == 0:
                self.logger.save_model(self.generator, f'generator_ckpt_{epoch}')
                self.logger.save_model(self.discriminator, f'discriminator_ckpt_{epoch}')

        return epoch_gen_loss.item(), epoch_dis_loss.item()       


    def calc_eval_losses(self, target_data, predicted_data):
        """
        Calculate evaluation losses.
        """
        # Convert PyG Data objects to tensors
        if isinstance(target_data, Data):
            target_data = target_data.x
        
        if isinstance(predicted_data, Data):
            predicted_data = predicted_data.x

        target_data = target_data.detach().cpu()
        predicted_data = predicted_data.detach().cpu()

        # L1 loss
        l1_loss = self.l1_loss(target_data, predicted_data).item()

        # Topological losses
        # _, cc_loss, bc_loss, ec_loss = self.topo_loss(target_data, predicted_data, return_components=True)
        _, ec_loss = self.topo_loss(target_data, predicted_data, return_components=True)

        # return l1_loss, cc_loss.item(), bc_loss.item(), ec_loss.item()
        return l1_loss, ec_loss.item()
        
    
    def eval(self, source_data, target_data, epoch):
        """
        Evaluate the GAN model on given data.
        """
        self.generator.eval()

        gen_outputs = []
        source_graphs = []
        target_graphs = []

        l1_losses = []
        # cc_losses = []
        # bc_losses = []
        ec_losses = []

        with torch.no_grad():
            for source_graph, target_graph in tqdm(zip(source_data, target_data), desc=f"Evaluating epoch {epoch}", total=len(source_data)):
                # Generator output
                source_graph = source_graph.to(self.device)
                gen_output = self.generator(source_graph)
                gen_outputs.append(gen_output.detach().cpu().numpy())

                # Evaluation losses
                # l1_loss, cc_loss, bc_loss, ec_loss = self.calc_eval_losses(target_graph, gen_output)
                l1_loss, ec_loss = self.calc_eval_losses(target_graph, gen_output)
                
                l1_losses.append(l1_loss)
                # cc_losses.append(cc_loss)
                # bc_losses.append(bc_loss)
                ec_losses.append(ec_loss)

                # Log source and target graphs
                source_graphs.append(source_graph.x.detach().cpu().numpy())
                target_graphs.append(target_graph.x.detach().cpu().numpy())

            # Calculate average losses for the epoch
            l1_losses = np.mean(l1_losses)
            # cc_losses = np.mean(cc_losses)
            # bc_losses = np.mean(bc_losses)
            ec_losses = np.mean(ec_losses)

            # Log epoch results
            self.logger.log_metric(l1_losses, epoch, 'Eval L1 loss')
            # self.logger.log_metric(cc_losses, epoch, 'Eval CC loss')
            # self.logger.log_metric(bc_losses, epoch, 'Eval BC loss')
            self.logger.log_metric(ec_losses, epoch, 'Eval EC loss')

            # print(f"Eval === L1 loss: {l1_losses:.3f} | CC loss: {cc_losses:.3f} | BC loss: {bc_losses:.3f} | EC loss: {ec_losses:.3f}")
            print(f"Eval === L1 loss: {l1_losses:.3f} | EC loss: {ec_losses:.3f}")

            return source_graphs, target_graphs, gen_outputs
    

    def run(self, source_train_data, target_train_data, source_test_data, target_test_data):
        """
        Train the GAN model and report performance on test dataset.
        """
        source_train_data, target_train_data = self.prepare_data(source_train_data, target_train_data)
        source_test_data, target_test_data = self.prepare_data(source_test_data, target_test_data)

        # DEBUG: Only run on 10 graphs
        # source_train_data = source_train_data[:10]
        # target_train_data = target_train_data[:10]

        # source_test_data = source_test_data[:10]
        # target_test_data = target_test_data[:10]

        # To log training losses
        gen_losses = []
        dis_losses = []

        # To log evaluation results
        eval_epochs = []
        source_matrix_test = []
        target_matrix_test = []
        output_matrix_test = []
        
        for epoch in range(1, self.n_epochs+1):
            # Train current epoch
            gen_loss, dis_loss = self.train_epoch(source_train_data, target_train_data, epoch)
            gen_losses.append(gen_loss)
            dis_losses.append(dis_loss)

            # Evaluate epoch
            if epoch % self.eval_freq == 0 or epoch == self.n_epochs:
                source, target, out = self.eval(source_test_data, target_test_data, epoch)

                eval_epochs.append(epoch)
                source_matrix_test.append(source)
                target_matrix_test.append(target)
                output_matrix_test.append(out)

        # Log source, target, and output graphs for the test dataset
        self.logger.save_numpy_array(eval_epochs, 'eval_epochs')
        self.logger.save_numpy_array(source_matrix_test, 'source_matrix_test')
        self.logger.save_numpy_array(target_matrix_test, 'target_matrix_test')
        self.logger.save_numpy_array(output_matrix_test, 'output_matrix_test') 

        # Create an animation showing evolution of results
        self.logger.animate_results(eval_epochs, source_matrix_test, target_matrix_test, output_matrix_test) 

        # Plot evaluation results from the final epoch for the first sample
        self.logger.plot_results(
            source_matrix_test[-1][0],
            target_matrix_test[-1][0],
            output_matrix_test[-1][0]
        ) 

        # Save final models
        self.logger.save_model(self.generator, 'generator')
        self.logger.save_model(self.discriminator, 'discriminator')

        # Clear CUDA cache
        torch.cuda.empty_cache() 
        
        # Close wandb run
        self.logger.finish()

