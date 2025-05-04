# fcvopt: Fractional cross-validation for hyperparameter optimization

This repository containts code to reproduce the results from the paper "Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms".

The experiments are all contained in the `experiments` folder. Each subdirectory within this folder contains scripts files to run each case study in the paper. Refer to the README file within each of the subdirectories for instructions to run the files.

## Installation 
Prior to running the experiments, the `fcvopt` package must be installed. This can be done through pip:

```{bash}
cd <path_to_directory>
pip install .[experiments]
```
This will also install the dependencies, along with the additional packages, such as SMAC and optuna, for running some of the experiments.

**Note:**
The experiments involving the SMAC algorithm require the `smac` library, which in turn requires the building and compliling the `pyrfr` package While the main functions of `fcvopt` do not depend on `pyrfr`, you might encounter build issues during its installation if you do not have a C++ compiler and the `swig` binary installed on your system. 


## Setting up virtual environments/ containers

For reproducibility, we provide two options for setting up the environment to run the experiments: a virtual environment using `venv` and a Docker container. 

### Virtual environment

The bash script file `venv_setup.sh` can be used to create a virtual environment and install the required packages. Ensure you have Python >= 3.8 and <=3.12 installed.

To run the script, use the following commands:

```{bash}
chmod +x venv_setup.sh
./venv_setup.sh
```

### Docker

The Dockerfile is provided to run the experiments in a container with the `fcvopt` package and all the required dependencies. The Dockerfile is based on the Python 3.10 debian image. To build the image, run the following command:

```{bash}
docker build -t fcvopt_test .
```

To run the container with the files in the `experiments` folder mounted, run the following command:

```{bash}
docker run -v <path_to_experiments_folder>:/app/experiments -it fcvopt_test
```

This will launch the container and open a bash shell. The experiments directory will be mounted in the container at `/app/experiments`. Mounting the directory allows you to access the files in the experiments folder from within the container, and any changes made to the files will be reflected in your local directory. Replace <path_to_experiments_folder> with the **absolute path** to your local experiments directory.  Relative paths will not work, as the container will not have access to your local file system. On Linux and MacOS, you can use the $(pwd) command to get the absolute path of the current directory. For example:

```{bash}
docker run -v $(pwd)/experiments:/app/experiments -it fcvopt_test
```


Once inside the container, you can navigate to the /app/experiments directory and run the experiments as needed. For example:

```{bash}
cd experiments
bash reproduce_rf.sh
```