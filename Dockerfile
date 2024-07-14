FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install the CPU only version of PyTorch
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install the fcvopt library along with required dependencies
# and the experiments extra dependencies to run the experiments
RUN pip install .[experiments]

# Set the default command to run when the container starts
ENTRYPOINT ["/bin/bash"]