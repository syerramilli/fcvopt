# fcvopt: Fractional cross-validation for hyperparameter optimization

This repository containts code to reproduce the results from the paper "Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms".

The experiements are all contained in the `experiments` folder. Each subdirectory within this folder contains scripts files to run each case study in the paper. Refer to the README file within each of the subdirectories for instructions to run the files. 

## Installation 
Prior to running them, the `fcvopt` package must be installed. This can be done through pip:

```{bash}
cd <path_to_directory>
pip install .[experiments]
```
This will also install the required packages, along with the additional packages required to run the experiments.