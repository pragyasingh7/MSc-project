import os
import hydra
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# from torch_geometric.data import DataLoader

from src.utils import calc_n_nodes_f, get_LR_from_HR
from src.trainers.gan_trainer import GANTrainer
# from src.dataset import KaggleDataset
from src.matrix_vectorizer import MatrixVectorizer


os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base="1.3.2", config_path="configs", config_name="experiment")
def main(config):
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Load dataset
    n_source_nodes = config.dataset.n_source_nodes
    n_target_nodes = config.dataset.n_target_nodes

    if config.dataset.dataset_name == 'kaggle':
        source_data = pd.read_csv(config.dataset.source_dir).to_numpy()
        target_data = pd.read_csv(config.dataset.target_dir).to_numpy()

    elif config.dataset.dataset_name == 'cropped_kaggle':
        matrix_vectorizer = MatrixVectorizer()
        target_data = pd.read_csv(config.dataset.target_dir).to_numpy()
        target_mat = [matrix_vectorizer.anti_vectorize(v, n_target_nodes) for v in target_data]
        source_mat = get_LR_from_HR(target_mat, scale=config.dataset.scale, pooling_type=config.dataset.pooling_type)
        source_data = np.array([matrix_vectorizer.vectorize(m) for m in source_mat])
    
    else:
        n_subjects = config.dataset.n_subjects
        n_source_nodes_f, n_target_nodes_f = calc_n_nodes_f(n_source_nodes, n_target_nodes)
        
        source_data = np.random.normal(0, 0.5, (n_subjects, n_source_nodes_f))
        target_data = np.random.normal(0, 0.5, (n_subjects, n_target_nodes_f))

    # Create object for KFold cross-validation
    kf = KFold(
        n_splits=config.experiment.kfold.n_splits,
        shuffle=config.experiment.kfold.shuffle,
        random_state=config.experiment.kfold.random_state
    )

    for fold, (train_index, test_index) in enumerate(kf.split(source_data)):
        # Source datasets
        source_train_data = source_data[train_index]
        source_test_data = source_data[test_index]

        # Target datasets
        target_train_data = target_data[train_index]
        target_test_data = target_data[test_index]

        # Run on limited data in debug mode
        if config.experiment.debug:
            source_train_data = source_train_data[:10]
            source_test_data = source_test_data[:10]
            target_train_data = target_train_data[:10]
            target_test_data = target_test_data[:10]

        # Create dataloader
        # train_dataset = KaggleDataset(source_train_data, target_train_data, n_source_nodes, n_target_nodes)
        # test_dataset = KaggleDataset(source_test_data, target_test_data, n_source_nodes, n_target_nodes)

        # train_dataloader = DataLoader(train_dataset, batch_size=config.experiment.batch_size, shuffle=True)
        # test_dataloader = DataLoader(test_dataset, batch_size=config.experiment.batch_size, shuffle=False)

        # Define and run trainer object for this fold
        trainer = GANTrainer(config, exp_tag=f'fold{fold+1}_')
        # trainer = IMANGraphNetTrainer(config, exp_tag=f'fold{fold}_')   
        trainer.run(source_train_data, target_train_data, source_test_data, target_test_data)

        print(f"Completed fold {fold+1}/{config.experiment.kfold.n_splits}")


if __name__ == "__main__":
    main()