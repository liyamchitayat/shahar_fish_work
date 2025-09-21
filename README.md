Zebrafish 5 dpf Segmentation (Cellpose on MIP)

This repo contains a simple script to compute Z‑max projections from ND2 files and segment cells with Cellpose, saving masks, overlays, and a composite PNG for quick QC.

Important: the main pipeline is MIP-only
- The primary script segments the Z‑max projection (2D). Filenames include `_zmax` to make this explicit.
- A separate helper enables 3D segmentation on TIFF stacks; it does not modify the MIP script.

Data are not committed (see `.gitignore`). Point the script at your local ND2 files.

Setup
- Python 3.9+ recommended
- Install dependencies:

```
pip install -r requirements.txt
```

The scripts use writable cache dirs for the Cellpose model and Matplotlib in the project folder (`cellpose_models/` and `.mplconfig/`). The first run will download ~1.1 GB Cellpose model weights.

Script (2D MIP)
- `scripts/segment_cellpose_nd2.py` (MIP-only)

Key behavior:
- Loads ND2, computes Z‑max projection of a specified channel
- Optional center crop for quick tests (`--crop 1024`)
- Runs Cellpose (v4 CPSAM) once per input and saves:
  - MIP (`projections/*.tif`)
  - Masks (`masks/*.tif`)
  - Overlay (`overlays/*_overlay*.png`)
  - Composite (original | masks-only | overlay) (`overlays/*_composite*.png`)
  - NPY artifacts in `npy/`
- Adds suffixes to avoid overwriting: `_crop{N}` or `_full`
- `--composite-only` builds composite PNG from existing MIP + masks without re-running Cellpose
- `--force` overwrites existing outputs

Channels to use
- imaging_data/fish3_Ty-GFP_bleached_Tyanti-26_8_25007.nd2 → channel 0
- imaging_data/fish10.nd2 → channel 0
- imaging_data/Ty-GFP_fixed_fish3001.nd2 → channel 1
- imaging_data/fish10_mKate-V5_V5anti-26_8_25004.nd2 → channel 1

Recommended hyperparameters (starting point)
These matched initial tests on the Ty-GFP fixed fish (round-ish cells):
- `--diameter 60`
- `--flowth 0.4`
- `--cellprob 0.0`
- `--rescale 1.0` (note: rescale is deprecated in v4; included here to document prior runs)

Tune `--diameter` if cells are under/oversegmented; increase `--cellprob` (e.g., 0.1–0.3) or `--flowth` (e.g., 0.6) to reduce false positives.

Usage examples (MIP)
1) Quick crop (1024×1024 center) to tune params without long runtime:

```
python3 scripts/segment_cellpose_nd2.py imaging_data/Ty-GFP_fixed_fish3001.nd2 \
  --channels 1 --outdir segmentation --diameter 60 --flowth 0.4 --cellprob 0.0 \
  --crop 1024
```

2) Build composite from existing crop outputs without re-running the model:

```
python3 scripts/segment_cellpose_nd2.py imaging_data/Ty-GFP_fixed_fish3001.nd2 \
  --channels 1 --outdir segmentation --composite-only --crop 1024
```

3) Full-image MIP + segmentation (CPU may be slow; consider a smaller `--crop` first):

```
python3 scripts/segment_cellpose_nd2.py imaging_data/Ty-GFP_fixed_fish3001.nd2 \
  --channels 1 --outdir segmentation --diameter 60 --flowth 0.4 --cellprob 0.0
```

4) Other files (replace channel accordingly):

```
python3 scripts/segment_cellpose_nd2.py imaging_data/fish10.nd2 \
  --channels 0 --outdir segmentation --diameter 60 --flowth 0.4 --cellprob 0.0

python3 scripts/segment_cellpose_nd2.py imaging_data/fish3_Ty-GFP_bleached_Tyanti-26_8_25007.nd2 \
  --channels 0 --outdir segmentation --diameter 60 --flowth 0.4 --cellprob 0.0

python3 scripts/segment_cellpose_nd2.py imaging_data/fish10_mKate-V5_V5anti-26_8_25004.nd2 \
  --channels 1 --outdir segmentation --diameter 60 --flowth 0.4 --cellprob 0.0
```

Notes
- Large ND2 files and all generated outputs are ignored by Git via `.gitignore`.
- If you need ND2 in Git, consider Git LFS; otherwise keep data local.
- CPU-only runs of CPSAM are slow; a GPU/MPS environment is recommended for full frames.
- The script sets `CELLPOSE_LOCAL_MODELS_PATH` to `cellpose_models/` in the project root so you don’t need admin permissions for model caching.

3D helper (TIFF stacks)
- `scripts/segment_cellpose_tiff_3d.py` runs true 3D on pre-extracted Z stacks (TIFF) using the same parameters (diameter, thresholds). It does not change the MIP script.

Examples (3D TIFF)
- Anisotropy = Z_um / XY_um_per_px. For fish10 data (XY ≈ 0.1696 µm/px; Z step ≈ 5 µm): anisotropy ≈ 29.5

```
CELLPOSE_LOCAL_MODELS_PATH="$(pwd)/cellpose_models" MPLCONFIGDIR="$(pwd)/.mplconfig" \
python3 scripts/segment_cellpose_tiff_3d.py segmentation3d/fish10-test_data_smaller.tif \
  --outdir segmentation3d --diameter 60 --cellprob 0.0 --flowth 0.4 --anisotropy 29.5

# Faster preview (512×512 center crop)
CELLPOSE_LOCAL_MODELS_PATH="$(pwd)/cellpose_models" MPLCONFIGDIR="$(pwd)/.mplconfig" \
python3 scripts/segment_cellpose_tiff_3d.py segmentation3d/fish10-test_data_smaller.tif \
  --outdir segmentation3d --diameter 60 --cellprob 0.0 --flowth 0.4 --anisotropy 29.5 --crop 512
```

Runtime guidance (CPU)
- 2D MIP 1024×1024: ~1–2 minutes
- 3D stack 6×669×708: ~1–3+ minutes (first run slower due to model load)
- Full 3D volumes are much slower on CPU; prefer GPU if available.

Cluster usage
- See `CLUSTER.md` for SLURM batch examples (2D MIP CPU/GPU, 3D TIFF CPU) and using node-local scratch for caches.

Reproducibility
- Parameters are recorded in your command line.
- The script stores masks as TIFF and optional flows/diameters as NPY in `segmentation/<sample>/npy/`.
