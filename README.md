# Summary
Implementation of Vecchia GPs for "Scalable Bayesian Optimization Using Vecchia Approximations of Gaussian Processes".

# Install 
It's assumed there is a conda environement with all dependencies which also has [conda compilers](https://anaconda.org/conda-forge/compilers).

To install, run `pip install .` inside the "code" folder. 

# code
Contains code for Vecchia GPs. 

# notebooks
Contains an exmaple using a Vecchia GP for a BO loop in [BoTorch](https://botorch.org/tutorials/turbo_1). 

# Dependencies
faiss 1.7.2, gpytorch 1.9.0, torch 1.12.1, numpy 1.23.3 and matplotlib 3.6.1.

# References
[arxiv paper](https://arxiv.org/abs/2203.01459).