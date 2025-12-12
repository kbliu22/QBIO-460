#!/bin/bash

# Manual environment creation with essential packages only

echo "Creating conda environment manually with essential packages..."

module purge
module load conda

# Create base environment with Python
conda create -n unikp_env python=3.8 -y

# Activate environment
source activate unikp_env

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install transformers and related packages
conda install -c conda-forge transformers -y
conda install -c conda-forge sentencepiece -y
conda install -c conda-forge protobuf -y

# Install other essential packages
conda install pandas numpy scipy scikit-learn -y

# Install any additional packages that might be needed
conda install -c conda-forge tokenizers -y

# Install any other packages from your original environment
# Check what else you need from: conda list -n unikp_env

echo ""
echo "Environment created successfully!"
echo ""
echo "To verify installation, run:"
echo "  source activate unikp_env"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available__()}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
