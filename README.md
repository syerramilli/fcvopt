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

## Setting up virtual environments/ containers

### Docker

A Dockerfile is provided to run the experiments in a container with the `fcvopt` package and all the required dependencies. The Dockerfile is based on the Python 3.10 debian image. To build the image, run the following command:

```{bash}
docker build -t fcvopt_test .
```

To run the container with the files in the `experiments` folder mounted, run the following command:

```{bash}
docker run -v <path_to_experiments_folder>:/app/experiments -it fcvopt_test
```

Replace <path_to_experiments_folder> with the absolute path to your local experiments directory. This will launch the container and open a bash shell. The experiments directory will be mounted in the container at `/app/experiments`. 

Once inside the container, you can navigate to the /app/experiments directory and run the experiments as needed. For example:

```{bash}
cd experiments
bash reprodce_rf.sh
```