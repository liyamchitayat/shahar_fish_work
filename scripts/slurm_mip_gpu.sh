#!/usr/bin/env bash
#SBATCH -J mip2d-gpu
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# module load cuda/12.x  # if needed on your cluster
# conda activate cellpose-mip

export CELLPOSE_LOCAL_MODELS_PATH="${SLURM_TMPDIR:-$PWD}/cellpose_models"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$PWD}/.mplconfig"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$MPLCONFIGDIR"

ND2=${ND2:-imaging_data/fish10.nd2}
CHAN=${CHAN:-0}
OUTDIR=${OUTDIR:-segmentation}

python3 scripts/segment_cellpose_nd2.py \
  "$ND2" \
  --channels "$CHAN" \
  --outdir "$OUTDIR" \
  --diameter 60 --flowth 0.4 --cellprob 0.0

