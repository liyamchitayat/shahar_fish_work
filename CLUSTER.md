Cluster Usage (SLURM examples)

Overview
- The main pipeline here is 2D MIP-only on ND2 (`scripts/segment_cellpose_nd2.py`).
- A separate helper runs 3D on TIFF stacks (`scripts/segment_cellpose_tiff_3d.py`).
- Prefer GPU nodes for speed. CPU-only runs of CPSAM v4 are slow, especially in 3D.

Environment
- Python 3.9+ with the packages in `requirements.txt`.
- If using Conda:
  - conda create -n cellpose-mip python=3.9
  - conda activate cellpose-mip
  - pip install -r requirements.txt
- Torch with CUDA: if your cluster requires a specific CUDA build, install the matching torch build per https://pytorch.org/get-started/locally/.

Caching
- Set model and Matplotlib caches to writable scratch to avoid $HOME limits and speed up:

```
export CELLPOSE_LOCAL_MODELS_PATH="$SLURM_TMPDIR/cellpose_models"
export MPLCONFIGDIR="$SLURM_TMPDIR/.mplconfig"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$MPLCONFIGDIR"
```

SLURM job: 2D MIP on ND2 (CPU example)

```
#!/usr/bin/env bash
#SBATCH -J mip2d
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p cpu
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# Activate your environment here
# module load anaconda
# conda activate cellpose-mip

export CELLPOSE_LOCAL_MODELS_PATH="${SLURM_TMPDIR:-$PWD}/cellpose_models"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$PWD}/.mplconfig"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$MPLCONFIGDIR"

python3 scripts/segment_cellpose_nd2.py \
  imaging_data/fish10.nd2 \
  --channels 0 \
  --outdir segmentation \
  --diameter 60 --flowth 0.4 --cellprob 0.0
```

SLURM job: 2D MIP on ND2 (GPU example)

```
#!/usr/bin/env bash
#SBATCH -J mip2d-gpu
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# module load cuda/12.x  # if needed
# conda activate cellpose-mip

export CELLPOSE_LOCAL_MODELS_PATH="${SLURM_TMPDIR:-$PWD}/cellpose_models"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$PWD}/.mplconfig"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$MPLCONFIGDIR"

python3 scripts/segment_cellpose_nd2.py \
  imaging_data/fish10.nd2 \
  --channels 0 \
  --outdir segmentation \
  --diameter 60 --flowth 0.4 --cellprob 0.0
```

SLURM job: 3D on TIFF stack (CPU example)

```
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

TIFF=segmentation3d/fish10-test_data_smaller.tif
ANISO=29.5  # Z/XY ratio (5 µm / 0.1696 µm/px)

python3 scripts/segment_cellpose_tiff_3d.py "$TIFF" \
  --outdir segmentation3d \
  --diameter 60 --cellprob 0.0 --flowth 0.4 \
  --anisotropy "$ANISO"
```

Array jobs
- If segmenting many ND2 files, use a job array and pass the file and channel via an index-mapped list or a TSV mapping.

Notes
- The MIP script is 2D-only; outputs include `_zmax` in filenames to emphasize this.
- Use `--crop` during parameter tuning to shorten runtime, then remove for full runs.
- Ensure logs/ exists or adjust `#SBATCH -o` path.
