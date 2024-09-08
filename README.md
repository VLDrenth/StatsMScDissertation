# Bayesian Batch Active Learning for Classification with Kernels

This repo contains code for my dissertation on Bayesian Batch Active Learning for Classification with Kernels. The project is structured as follows:

1. **Data**: This directory contains the datasets used in the dissertation. Each dataset is stored in a separate subdirectory.

2. **Code**: This directory contains all the code files for the project. It is further organized into the following subdirectories:
    - **Python scripts**:
        - `prepare_data`: Contains scripts for creating the data loaders.
        - `models`: Contains implementations of the CNN and MLP used in experiments.
        - `active_learning`: Contains function that implements the main active learning logic, based on code from [BlackHC/batchbald_redux](https://github.com/BlackHC/batchbald_redux).
        - `badge`: Contains implementation of BADGE, based on code from [BlackHC/active_learning_redux](https://github.com/BlackHC/active_learning_redux/)
        - `approximation`: Contains the computation of the similarity matrix and its feature map.
        - `bald_sampling`: Contains LA based implementation of BALD and the approximation of the joint MI.
        - `laplace_batch`: Contains implementation of batch acquisition of all used methods.
        - `training_models`: Contains implementation to train models.
        - `utils`: Contains some utility functions used throughout the project.
    - **Python Notebooks**:
        - `run_experiments`: Implements the full experiment.
        - `plotting`: Used for plotting the accuracy and loss curves
        - `running_times`: Used for plotting the run-time plot
        - `statistical_test`: Used for generating the results for the Friedman and Nemenyi tests.




