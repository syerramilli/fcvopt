#!/bin/bash

# Set up a virtual environment
# Usage: ./venv_setup.sh [venv_name]
# Default venv name: .venv
VENV_NAME=${1:-.venv}

python3 -m venv "$VENV_NAME"

# Activate the virtual environment
source "$VENV_NAME/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install the CPU only version of PyTorch (separate index_url needed for Linux)
pip install torch==2.7 --index-url https://download.pytorch.org/whl/cpu

# Install the fcvopt library along with required dependencies
# and the experiments extra dependencies to run the experiments
pip install .[experiments]

echo "Setup complete. To activate the virtual environment, run 'source $VENV_NAME/bin/activate'."
