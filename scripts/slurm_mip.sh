#!/usr/bin/env bash
#SBATCH -J mip2d
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p cpu
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# Activate your environment
# module load anaconda
# conda activate cellpose-mip

# Use node-local scratch for caches
export CELLPOSE_LOCAL_MODELS_PATH="${SLURM_TMPDIR:-$PWD}/cellpose_models"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$PWD}/.mplconfig"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$MPLCONFIGDIR"

# Inputs (edit as needed)
ND2=${ND2:-imaging_data/fish10.nd2}
CHAN=${CHAN:-0}
OUTDIR=${OUTDIR:-segmentation}

python3 scripts/segment_cellpose_nd2.py \
  "$ND2" \
  --channels "$CHAN" \
  --outdir "$OUTDIR" \
  --diameter 60 --flowth 0.4 --cellprob 0.0

