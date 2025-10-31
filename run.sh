#!/bin/bash
# This script runs the entire classification pipeline.
# Designed for use on ETH Zurich's Euler cluster with GPU support.
# You may need to adjust the module load commands based on your environment.

#SBATCH --job-name=classification
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=48:00:00
#SBATCH --output=logs/classification_%j.log
#SBATCH --error=logs/classification_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs
set -euo pipefail

dir="$SCRATCH/llm-flow"
export HF_HOME="$dir/.hf/"
export UV_CACHE_DIR="$dir/.uv/"
export UV_PROJECT_ENVIRONMENT="$dir/.venv"

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --quiet

# echo "Beginning classification at $(date)"

# python -m src.llm_flow \
#   flow=sentiment \
#   output_name=sentiment

# echo "run finished at $(date)"
