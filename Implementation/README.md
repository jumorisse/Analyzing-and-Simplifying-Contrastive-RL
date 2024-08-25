# Implementation

Based on https://github.com/google-research/google-research/tree/master/contrastive_rl

This directory contains the code used to run the experiments for my Master thesis.
My implementation is based on that of Eysenbach et al. (2023) and, therefore, this codebase largely consists of their code available at https://github.com/google-research/google-research/tree/master/contrastive_rl .
I list my own files and the files I modified below and give instructions on how to reproduce my results.

## Files Developed

These files I wrote myself:

- **`experiment_01.py`**: Runs my first experiment, i.e. produces visualizations of CRL's representation space and critic function.
- **`experiment_02.py`**: Runs my second experiment, i.e. compares greedy action selection to that of the parameterized actor.
- **`experiment_03.py`**: Evaluates my simplifications, i.e. plots learning curves and performs an evaluation of greedy vs. original CRL. This script does not actually perform the training, but just plots it and evaluates the resulting agents. For instruction on reproducing training, see below.
- **`appendix.py`**: Produces the figures of my appendix.
- **`experiment_utils.py`**: Contains functions used throughout my experiment scripts.
- **`greedy_utils.py`**: Contains functions used for evaluating greedy CRL.

## Important Modified Files

These are files also contained in the implementation by Eysenbach et al. (2023) but which I made modifications to in order to allow my experiments:

- **`lp_contrastive.py`**: This file is used to train CRL. I made changes to it for allowing the use a greedy or random actor.
- **`lp_contrastive_only_sa.py`**: This is a copy of lp_contrastive.py but for my single-encoder approach.
- **`contrastive/networks.py`**: In this script, CRL's networks (policy, sa-encoder, g-encoder, critic) are defined. I made changes to it to allow the use of a greedy actor during CRL training.
- **`contrastive/utils.py`**: This script contains actor classes used in other scripts. I added two classes: RandomActor and GreedyAcotr.
- **`contrastive/learning.py`**: This script defines the learning procedure for CRL. I made changes to it in order to skip policy evaluation when using a greedy actor.

## Instructions for Reproduction

To reproduce the results from the experiments, follow the steps below.

### 1. Creating the Conda Environment

For my implementation, I use conda environments.
Two environments are needed, one to train CRL and one to perform the experiments.
The requirements for the training environment are given in contrastive_rl_requirements.txt.
The requirements for the experiments are given in experiment_requirements.txt.

### 2. Training original CRL or one of Our Versions (Greedy CRL or Single-Encoder CRL)

You can either use the training logs and checkpoints produced by me or you have to train CRL yourself.
If you train yourself, you have to activate your training environment and run either lp_contrastive.py (for original CRL, greedy CRL, or the random actor baseline) or lp_contrastive_only_sa.py (for the Single-Encoder approaches).
When training, network parameters will periodically be saved to manual_checkpoints/latest. 
Additionally, training logs are saved into your local acme directory.
After training, the parameters and logs need to be put into the corresponding directory under manual_checkpoints.
For example, when training greedy CRL with 25 actions and seed 42, they need to be put into manual_checkpoints/two_encoders/point_Spiral11x11/greedy_randominit_25actions_seed42/ .
New directories than the existing ones can be created but require to change paths within the experiment files.

### 3. Reproducing experiments.
To reproduce my experiments, the corresponding experiment files can simply be run from within the experiment environment.
This requires all training results necessary are stored under the right paths in manual_checkpoints/ .
