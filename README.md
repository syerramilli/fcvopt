# fcvopt: Fractional cross-validation for hyperparameter optimization

This repository containts code to reproduce the results from the paper "Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms".

The experiements are all contained in the `experiments` folder. Each subdirectory within this folder contains scripts files to run each case study in the paper. Refer to the README file within each of the subdirectories for instructions to run the files. 

## Installation 
Prior to running them, the `fcvopt` package must be installed. This can be done through pip:

```{bash}
pip install <path_to_directory>
```

This will also install any required packages.

**Requirements**:
- python >= 3.8
- torch >= 1.13
- gpytorch >= 1.9
- botorch >= 0.8
- numpy >= 1.2
- scipy >= 1.10
- scikit-learn >= 1.12
- skorch >= 0.13