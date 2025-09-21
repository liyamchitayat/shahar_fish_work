#!/usr/bin/env bash
#SBATCH -J cellpose3d
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p cpu
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# conda activate cellpose-mip

export CELLPOSE_LOCAL_MODELS_PATH="${SLURM_TMPDIR:-$PWD}/cellpose_models"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$PWD}/.mplconfig"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$MPLCONFIGDIR"

TIFF=${TIFF:-segmentation3d/fish10-test_data_smaller.tif}
ANISO=${ANISO:-29.5}  # Z_um / XY_um_per_px (e.g., 5 / 0.1696)
OUTDIR=${OUTDIR:-segmentation3d}

python3 scripts/segment_cellpose_tiff_3d.py "$TIFF" \
  --outdir "$OUTDIR" \
  --diameter 60 --cellprob 0.0 --flowth 0.4 \
  --anisotropy "$ANISO"

