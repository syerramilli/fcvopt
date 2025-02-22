FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# Note: The .dockerignore file is used to exclude certain files and directories 
# from being copied. This includes the experiments directory which is not needed
# for installing the library. It is recommnded to mount the experiments directory
# as a volume when running the container.
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install the CPU only version of PyTorch (the index-url must be specified for Linux distributions)
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install the fcvopt library along with required dependencies
# and the extra dependencies for the experiments
RUN pip install .[experiments]

# Set the default command to run when the container starts
ENTRYPOINT ["/bin/bash"]