#!/bin/bash

# Set up a virtual environment
python3 -m venv fcvopt_test

# Activate the virtual environment
source fcvopt_test/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the CPU only version of PyTorch (separate index_url needed for Linux)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install the fcvopt library along with required dependencies
# and the experiments extra dependencies to run the experiments
pip install .[experiments]

echo "Setup complete. To activate the virtual environment, run 'source fcvopt_test/bin/activate'."