#!/bin/bash
# =============================================================================
# U-VLM Environment Setup Script
# =============================================================================
# Creates a complete conda environment for U-VLM training, inference, and evaluation.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Configuration:
#   Set environment variables before running this script to customize paths:
#     CONDA_SOURCE   - Path to conda.sh (default: auto-detect from $CONDA_PREFIX)
#     CONDA_ENV_DIR  - Directory for conda environments (default: $HOME/.conda/envs)
#     ENV_NAME       - Name of the conda environment (default: uvlm)
#     NNUNET_DIR     - Path to nnUNet source code (default: ../nnUNet)
#     UVLM_DIR       - Path to U-VLM source code (default: current directory)
#     PYTHON_VERSION - Python version (default: 3.10)
#
# Examples:
#   # Default setup
#   bash scripts/setup_env.sh
#
#   # Custom setup
#   ENV_NAME=my_uvlm NNUNET_DIR=/path/to/nnUNet bash scripts/setup_env.sh
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# 0. Auto-detect and configure paths
# ---------------------------------------------------------------------------
CONDA_SOURCE="${CONDA_SOURCE:-}"
ENV_NAME="${ENV_NAME:-uvlm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
NNUNET_DIR="${NNUNET_DIR:-}"
UVLM_DIR="${UVLM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/.conda/envs}"

# Auto-detect conda source if not set
if [ -z "$CONDA_SOURCE" ]; then
    # Try common conda locations
    for candidate in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/usr/local/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$candidate" ]; then
            CONDA_SOURCE="$candidate"
            break
        fi
    done
fi

if [ -z "$CONDA_SOURCE" ] || [ ! -f "$CONDA_SOURCE" ]; then
    echo "ERROR: Cannot find conda.sh. Please set CONDA_SOURCE environment variable."
    echo "  Example: CONDA_SOURCE=/path/to/miniconda3/etc/profile.d/conda.sh bash scripts/setup_env.sh"
    exit 1
fi

# Auto-detect nnUNet directory if not set
if [ -z "$NNUNET_DIR" ]; then
    for candidate in \
        "$(dirname "$UVLM_DIR")/nnUNet" \
        "$(dirname "$UVLM_DIR")/nnUNet-master" \
        "$HOME/nnUNet"; do
        if [ -d "$candidate" ]; then
            NNUNET_DIR="$candidate"
            break
        fi
    done
fi

echo "=============================================="
echo "U-VLM Environment Setup"
echo "=============================================="
echo "CONDA_SOURCE : $CONDA_SOURCE"
echo "ENV_NAME     : $ENV_NAME"
echo "PYTHON       : $PYTHON_VERSION"
echo "CONDA_ENV_DIR: $CONDA_ENV_DIR"
echo "NNUNET_DIR   : $NNUNET_DIR"
echo "UVLM_DIR     : $UVLM_DIR"
echo "=============================================="

source "$CONDA_SOURCE"
# Unset conflicting aliases before setting env dirs
unset CONDA_ENVS_PATH
export CONDA_ENVS_DIRS="$CONDA_ENV_DIR"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Creating conda environment: $ENV_NAME"
conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION -c pytorch -c nvidia
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 2. Install PyTorch via pip (avoids MKL compatibility issues with conda PyTorch)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Installing PyTorch..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 3. Install core dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installing core dependencies..."
pip install --no-cache-dir \
    "batchgenerators>=0.25" \
    "scikit-learn>=1.0.0" \
    "scipy>=1.7.0" \
    "pandas>=1.3.0" \
    "numpy>=1.24"

# Medical imaging
pip install --no-cache-dir \
    "blosc2>=3.0.0" \
    "SimpleITK>=2.2.0" \
    "nibabel>=4.0.0" \
    "pydicom>=2.3.0"

# Architecture support
pip install --no-cache-dir \
    "dynamic-network-architectures>=0.4.0" \
    "acvl-utils>=0.2.0"

# LLM
pip install --no-cache-dir \
    "tokenizers>=0.13.0" \
    "transformers>=4.36.0" \
    "peft>=0.7.0" \
    "accelerate>=0.25.0"

# Utilities
pip install --no-cache-dir \
    "tqdm>=4.64.0" \
    "pyyaml>=6.0" \
    "einops>=0.6.0" \
    "scikit-image>=0.19.0" \
    "matplotlib" \
    "seaborn>=0.12.0" \
    "imagecodecs" \
    "yacs" \
    "pycocoevalcap"

# ---------------------------------------------------------------------------
# 4. Install nnUNet from source
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Installing nnUNet..."
if [ -n "$NNUNET_DIR" ] && [ -d "$NNUNET_DIR" ]; then
    echo "Installing nnUNet from: $NNUNET_DIR"
    cd "$NNUNET_DIR"
    pip install -e . --no-deps
else
    echo "WARNING: nnUNet source directory not found. Skipping nnUNet installation."
    echo "  Set NNUNET_DIR to the nnUNet source directory and re-run."
    echo "  Alternatively, install manually: pip install nnunetv2"
fi

# ---------------------------------------------------------------------------
# 5. Install U-VLM from source
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Installing U-VLM..."
cd "$UVLM_DIR"
pip install -e . --no-deps
pip install -e .

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
python -c "
import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import nnunetv2; print(f'nnUNet v2: {nnunetv2.__version__ if hasattr(nnunetv2, \"__version__\") else \"installed\"}')
import uvlm; print(f'U-VLM: installed')
import blosc2; print(f'Blosc2: {blosc2.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
import SimpleITK as sitk; print(f'SimpleITK: {sitk.Version()}')

# Verify entry points
import pkg_resources
for ep in ['uvlm_train', 'uvlm_inference', 'uvlm_evaluate']:
    try:
        pkg_resources.load_entry_point('uvlm', 'console_scripts', ep)
        print(f'Entry point {ep}: OK')
    except Exception as e:
        print(f'Entry point {ep}: {e}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "=============================================="
