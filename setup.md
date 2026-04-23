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
git clone <your-repository-url>
cd ae598-arp-fp
```

### 2. Initialize Submodules (if using submodules)

If the embedded repositories have been converted to Git submodules, initialize them:

```bash
git submodule update --init --recursive
```

This will fetch IsaacLab, NaVILA, and legged-loco repositories.

## Environment Setup

### Option 1: Setup Both Environments

Run the setup script to create both environments automatically:

```bash
# From the project root directory
conda env create -f navila_env.yml
conda env create -f isaaclab_env.yml
```

### Option 2: Setup Individually

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

### 3. CPU-Only Fallback

If GPU setup is problematic, both environments can run on CPU (though it will be slower):

```bash
# The environments are configured to work on CPU by default
# GPU acceleration is automatic if available
```

## Post-Installation Setup

### For NaVILA Environment

After creating the environment, run any build or installation steps:

```bash
conda activate navila

# Example build commands - adjust based on your actual needs
# cd NaVILA
# pip install -e .
# python setup.py develop
```

### For IsaacLab Environment

```bash
conda activate isaaclab

# Example build commands - adjust based on your actual needs
# cd IsaacLab
# python setup.py develop
# ./python.sh -m pip install -e .
```

> **Note**: Replace the example commands above with the actual build steps required for your projects. Common patterns include `pip install -e .` (editable install) or `python setup.py develop`.

## Activating Environments

Once environments are created, activate them as needed:

```bash
# Activate NaVILA environment
conda activate navila

# Activate IsaacLab environment
conda activate isaaclab

# Deactivate current environment
conda deactivate
```

## Verification

### Verify Environment Creation

```bash
conda env list
```

You should see both `navila` and `isaaclab` listed.

### Verify Package Installation

```bash
conda activate navila
python -c "import <key-package>"  # Replace with a key package from your yaml

conda activate isaaclab
python -c "import <key-package>"  # Replace with a key package from your yaml
```

### Run Basic Tests (if available)

```bash
# Test NaVILA
conda activate navila
python -c "from <module> import <class>; print('NaVILA imported successfully')"

# Test IsaacLab
conda activate isaaclab
python -c "from <module> import <class>; print('IsaacLab imported successfully')"
```

## Directory Structure

```
ae598-arp-fp/
├── setup.md                    # This file
├── navila_env.yml             # NaVILA conda environment
├── isaaclab_env.yml           # IsaacLab conda environment
├── NaVILA/                    # NaVILA repository (submodule or embedded)
├── IsaacLab/                  # IsaacLab repository (submodule or embedded)
├── legged-loco/               # Legged locomotion repo (submodule or embedded)
└── [other project files]
```

## Troubleshooting

### Issue: "Command not found: conda"

**Solution**: Conda is not in your PATH. Either:
- Reinstall Conda/Mamba with PATH setup enabled
- Manually add Conda to your PATH:
  ```bash
  export PATH="$HOME/miniconda3/bin:$PATH"
  ```

### Issue: Package conflicts when creating environment

**Solution**: Use Mamba instead of Conda (faster and better conflict resolution):
```bash
conda install mamba -c conda-forge
mamba env create -f navila_env.yml
mamba env create -f isaaclab_env.yml
```

### Issue: CUDA not found / GPU not detected

**Solution**: 
1. Verify NVIDIA driver: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   conda activate isaaclab
   conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

### Issue: "Submodule not found" errors

**Solution**: If using submodules, ensure they're initialized:
```bash
git submodule update --init --recursive
```

Or fetch them manually:
```bash
cd NaVILA && git clone <isaaclab-url> .
cd ../IsaacLab && git clone <navila-url> .
cd ../legged-loco && git clone <legged-loco-url> .
```

### Issue: Permission denied errors on Linux/Mac

**Solution**: Ensure you have read/write permissions:
```bash
chmod -R u+w .
```

### Issue: Environment size too large

**Solution**: Clean conda cache to free space:
```bash
conda clean --all
mamba clean --all  # if using mamba
```

## Useful Commands

### List all environments

```bash
conda env list
```

### Remove an environment

```bash
conda env remove --name navila
conda env remove --name isaaclab
```

### Export current environment (to create updated yaml)

```bash
conda activate navila
conda env export > navila_env_updated.yml
```

### Update environment from yaml

```bash
conda env update --file navila_env.yml --prune
```

### Check environment info

```bash
conda activate navila
conda info
```

## Additional Resources

- [Conda Documentation](https://docs.conda.io/)
- [Mamba Documentation](https://mamba.readthedocs.io/)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the error message carefully
2. Search for the error in the project's issue tracker
3. Consult the documentation for IsaacLab and NaVILA
4. Create a detailed issue report with:
   - Your OS and hardware specs
   - Output of `conda info`
   - Full error traceback
   - Steps to reproduce

---

**Last Updated**: April 2026  
**Tested on**: Ubuntu 20.04 LTS, CUDA 12.1, Python 3.10+
