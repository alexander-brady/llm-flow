#!/bin/bash
#SBATCH --job-name=old_news
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=48:00:00
#SBATCH --output=logs/old_news_%j.log
#SBATCH --error=logs/old_news_%j.err
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

VENV_PATH="$SCRATCH/classification/.venv"

export HF_HOME="$SCRATCH/classification/cache"

# RESET_ENV can also be set to "true" to force a fresh environment
if [ ! -d "$VENV_PATH" ]; then
  RESET_ENV="true"
fi

if [ "$RESET_ENV" == "true" ]; then
  rm -rf "$VENV_PATH"
  python3 -m venv "$VENV_PATH"
  echo "Virtual environment created at $VENV_PATH at $(date)"
fi

source "$VENV_PATH/bin/activate"
 
if [ "$RESET_ENV" == "true" ]; then
    pip install --upgrade --quiet pip
    pip install --upgrade --quiet -r requirements.txt
    echo "Dependencies installed at $(date)"
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

echo "Beginning classification at $(date)"

python -m src.llm_flow \
  output_name=old_news \
  flow=old_news

echo "run finished at $(date)"