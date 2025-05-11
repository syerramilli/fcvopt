# Notes on reproducibility

This directory contains the code and data necessary to reproduce the results in the paper. The code is written in Python and it is assumed that the `fcvopt` library along with the required libraries in the `../requirements.txt` file are installed. Additionally, the following libraries are required:
- for generating the plots: `matplotlib` and `seaborn`. 
- for running the xgboost example: `xgboost` and `pandas`

The experiments take a while to run since fitting the supervised learning model can be expensive, and the experiments are run 15 times for each optimization algorithm. 

To expedite the reproducibility requirement for publication, we provide the bash script file `reproduce_rf.sh` that can be used to reproduce the results for tuning the hyperparameters of the random forest classifications models (Figure 3 in the paper). Optimization scripts for this case are in the `rf_high_dim` directory. Run this script from the command line as follows:

```{bash}
bash reproduce_rf.sh 
```

By default, the script will run the experiments and generate figures in the .png and .pdf format for the `madelon` dataset (Figure 3(a) in the paper). These will be saved in the `figures_auto` folder as `fig-rf-madelon.png` and `fig-rf-madelon.pdf` respectively. To generate the figures for the other two datasets, change the `DATASET` variable in the `reproduce_rf.sh` file from `madelon` to either `gina` or `bioresponse`. 

**Note about running times**:
The script will take a while to run, as it will run all 4 optimization algorithms 15 times, each time with a different random seed. The scripts are set up to use all available cores. The running time will depend on the number of cores available on your machine.