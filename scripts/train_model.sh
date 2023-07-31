#!/bin/sh
#SBATCH --gres=gpu:1 --time=1-00:00:00
#SBATCH --output=/home/myurachinskiy/YSDA/ML2/ml2-papers/scripts/outs/slurm-%j.out
nvidia-smi
date

PY_PATH="/home/myurachinskiy/YSDA/ML2/ml2-papers/papers/train_model.py"
ML_PAPERS_PATH="/home/myurachinskiy/YSDA/ML2/ml2-papers"

export PYTHONPATH="${PYTHONPATH}:${ML_PAPERS_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd $ML_PAPERS_PATH || exit
python -u $PY_PATH
# kernprof -l $PY_PATH
