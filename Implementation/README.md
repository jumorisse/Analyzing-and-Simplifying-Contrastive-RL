# Implementation

This directory contains the code used to run the experiments for my Master thesis.
My implementation is based on that of [Eysenbach et al. (2023)](https://github.com/google-research/google-research/tree/master/contrastive_rl).
Therefore, this codebase largely consists of their code.
Below, I list the files I developed and the ones I modified, along with instructions on how to reproduce my results.

## Files Developed

These are the files I added to the original CRL implementation:

- **`experiment_01.py`**: Runs my first experiment, producing visualizations of CRL's representation space and critic function.
- **`experiment_02.py`**: Runs my second experiment, comparing greedy action selection to that of the parameterized actor.
- **`experiment_03.py`**: Evaluates my simplifications by plotting learning curves and performing an evaluation of greedy vs. original CRL. This script does not actually perform the training but just plots the results and evaluates the resulting agents. For instructions on reproducing the training, see below.
- **`appendix.py`**: Produces the figures for the appendix of my thesis.
- **`experiment_utils.py`**: Contains utility functions used across my experiment scripts.
- **`greedy_utils.py`**: Contains functions for evaluating the greedy CRL approach.

## Important Modified Files

These are files originally included in the implementation by Eysenbach et al. (2023), which I modified to enable my experiments:

- **`lp_contrastive.py`**: This file is used to train CRL. I made changes to allow the use of a greedy or random actor.
- **`lp_contrastive_only_sa.py`**: A modified version of `lp_contrastive.py` for my single-encoder approach.
- **`contrastive/networks.py`**: Defines CRL's networks (policy, sa-encoder, g-encoder, critic). I modified this file to allow the use of a greedy actor during CRL training.
- **`contrastive/utils.py`**: Contains actor classes used in other scripts. I added two classes: `RandomActor` and `GreedyActor`.
- **`contrastive/learning.py`**: Defines the learning procedure for CRL. I made changes to skip policy evaluation when using a greedy actor.

## Instructions for Reproduction

To reproduce the results from the experiments, follow the steps below.

### 1. Creating the Conda Environment

For my implementation, I use Conda environments. Two environments are needed: one for training CRL and one for performing the experiments.

- **Training Environment:**  
  Install the required packages using the `contrastive_rl_requirements.txt` file.

- **Experiment Environment:**  
  Install the required packages using the `experiment_requirements.txt` file.

### 2. Training Original CRL or One of Our Versions (Greedy CRL or Single-Encoder CRL)

You can either use the training logs and checkpoints produced by me or train CRL yourself. To train, activate your training environment and run either `lp_contrastive.py` (for original CRL, greedy CRL, or the random actor baseline) or `lp_contrastive_only_sa.py` (for the Single-Encoder approaches).

When training, network parameters will periodically be saved to `manual_checkpoints/latest`. Additionally, training logs are saved to your local `acme` directory. After training, the parameters and logs need to be placed in the corresponding directory under `manual_checkpoints`.

For example, when training greedy CRL with 25 actions and seed 42, they should be placed in:

`manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42/`

New directories can be created as needed but may require updating paths within the experiment files.

### 3. Reproducing Experiments

To reproduce my experiments, the corresponding experiment files can simply be run from within the experiment environment. This requires that all necessary training results are stored in the correct paths under `manual_checkpoints/`.

