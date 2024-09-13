# MSc-project

Code submission for MSc project work on 'Generative GNNs for Graph Super-resolution'.

In this project, we propose two novel frameworks to tackle existing limitations of graph super-resolution models: (1) They use simple linear algebraic trick to increase the number of obtain node features for high-resolution (HR) graphs from low-resolution(HR) graphs. This violates the graph structure of the problem. (2) GNNs based on node representation learning has limited capacity to model edge features. 

To solve these, we propose two frameworks: (1) Bi-SR (Bipartite Graph Super-Resolution), and (2) DEFEND (Dual Graphs for Edge Feature learning and Detection). We perform extensive theoretical and empirical analysis to evaluate these frameworks. 

The experiments are done across three sets of datasets: (1) Physics-inspired dummy dataset, (2) Simulated datasets using traditional graph generation methods, and (3) Functional Connectome super-resolution. Please follow below steps to reproduce these experiments:

### Set-up

1. Create virtual environment for the project:

    ```python3 -m venv venv```
2. Activate virtual environment:

    ```source venv/bin/activate```
3. Install packages


    ```pip install -r requirements.txt```


### Physics-inspired dummy dataset

The configurations for these experiments are stored under  `configs/physics.py`. Please change them as required and run:

```
python3 runs/physics.py
```

### Simulated datasets

The configurations for these experiments are stored under  `configs/simulated.py`. Please specify the correct `DATASET_TYPE` there and change other configs as required before running:

```
python3 runs/simulated.py
```

### Brain Graph Dataset

The configurations for these experiments are stored under  `configs/brain.py`. Please change the configs as required before running:
```
python3 runs/brain/main.py
```

To run our adapted version of IMANGraphNet, please use:

```
python3 runs/brain/iman_adpated.py
```

To run our proposed AutoEncoder baseline, please use:

```
python3 runs/brain/autoencoder.py
```
