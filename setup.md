# ARP Final Project Setup Guide

This guide provides step-by-step instructions for setting up the NaVILA and IsaacLab conda environments for the ARP Final Project.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Conda** or **Mamba** (recommended for faster package resolution)
  - [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - [Install Mamba](https://mamba.readthedocs.io/en/latest/installation.html)
- **Git** (for cloning the repository)
- **GPU (Optional but recommended)**
  - NVIDIA GPU with CUDA capability
  - [NVIDIA Driver](https://www.nvidia.com/download/driverDetails.aspx)
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (v11.8 or v12.1 recommended)
  - [cuDNN](https://developer.nvidia.com/cudnn)

## Repository Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ganesh1102/ae598-arp-fp
cd ae598-arp-fp
```

### 2. Initialize Submodules (if using submodules)

If the embedded repositories have been converted to Git submodules, initialize them:

```bash
git submodule update --init --recursive
```

This will fetch IsaacLab, NaVILA, and legged-loco repositories.

## Environment Setup

#### Setup NaVILA Environment

```bash
conda env create -f navila_env.yml
conda activate navila
```

#### Setup IsaacLab Environment

```bash
conda env create -f isaaclab_env.yml
conda activate isaaclab
```

To finish set up, follow step 4 onwards at the official legged-loco repo: https://github.com/yang-zj1026/legged-loco
You can find test code at the linked URL. If running on headless server, make sure to add "--headless".

## GPU/CUDA Configuration

If you have an NVIDIA GPU and want to use it:

### 1. Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

Both commands should return version information.

### 2. Verify GPU Access in Conda Environments

After activating an environment, check GPU availability:

```bash
conda activate navila  # or isaaclab
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"  # if available
```

If `False` is returned for the first command, check:
- Your CUDA/cuDNN installation
- Your conda environment's PyTorch version matches your CUDA toolkit
- Your GPU drivers are up to date