# Notes on reproducibility

This directory contains the code and data necessary to reproduce the results in the paper. The code is written in Python and it is assumed that the `fcvopt` library along with the required libraries in `requirement.txt` are installed. Additionally, the following libraries are required:
- for generating the plots: `matplotlib` and `seaborn`. 
- for running the xgboost example: `xgboost` and `pandas`

The experiments take a while to run since fitting the supervised learning model can be expensive, and the experiments are run 15 times for each optimization algorithm. 

To expedite the reproducibility requirement for publication, we provide the bash script file `reproduce_rf.sh` that can be used to reproduce the results for tuning the hyperparameters of the random forest classifications models (Figure 3 in the paper). ptimization scripts for this case are in the `rf_high_dim` directory. Run this script from the command line as follows:

```{bash}
bash reproduce_rf.sh 
```

By default, the script will run the experiments and generate the figure for the `madelon` dataset. To generate the figures for the other two datasets, change the `DATASET` variable in the `reproduce_rf.sh` file from `madelon` to either `gina` or `bioresponse`.