#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# Writable caches (set BEFORE importing cellpose/matplotlib)
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(Path.cwd() / "cellpose_models"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import numpy as np
import tifffile as tiff
from cellpose import models
from cellpose import plot as cp_plot
import matplotlib.pyplot as plt


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max() or 1.0)
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return img


def save_qc_mips(stack: np.ndarray, masks3d: np.ndarray, out_png: Path):
    img_mip = stack.max(axis=0)
    msk_mip = (masks3d > 0).max(axis=0).astype(np.uint8)
    img = normalize_for_display(img_mip)
    overlay = cp_plot.mask_overlay(img, msk_mip)
    mask_rgb = cp_plot.mask_rgb(msk_mip)
    orig_rgb = np.dstack([img, img, img])
    def to_float01(a):
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        return np.clip(a, 0, 1)
    comp = np.concatenate([to_float01(orig_rgb), to_float01(mask_rgb), to_float01(overlay)], axis=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png.as_posix(), comp)


def main():
    ap = argparse.ArgumentParser(description="3D Cellpose on a TIFF stack (Z,Y,X) using original parameters.")
    ap.add_argument("tiff_path", help="Path to 3D TIFF stack (Z,Y,X)")
    ap.add_argument("--outdir", default="segmentation3d_tiff", help="Output directory root")
    # Original parameters
    ap.add_argument("--diameter", type=float, default=60.0)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--flowth", type=float, default=0.4)
    ap.add_argument("--anisotropy", type=float, default=None, help="Z/XY size ratio; if None, assume 1.0")
    ap.add_argument("--crop", type=int, default=None, help="Optional center crop size (pixels)")
    args = ap.parse_args()

    path = Path(args.tiff_path)
    stack = tiff.imread(path.as_posix())
    if stack.ndim != 3:
        raise SystemExit("Expect (Z,Y,X) stack in TIFF")

    if args.crop is not None:
        Z, H, W = stack.shape
        s = min(args.crop, H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        stack = stack[:, y0:y0+s, x0:x0+s]

    out_root = Path(args.outdir) / path.stem
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "overlays").mkdir(parents=True, exist_ok=True)

    print(f"Running 3D Cellpose on {path.name} with diameter={args.diameter}, cellprob={args.cellprob}, flowth={args.flowth}")
    model = models.CellposeModel(gpu=False)
    result = model.eval(
        stack,
        do_3D=True,
        z_axis=0,
        diameter=args.diameter,
        cellprob_threshold=args.cellprob,
        flow_threshold=args.flowth,
        anisotropy=(args.anisotropy if args.anisotropy is not None else 1.0),
        normalize=True,
        augment=False,
        compute_masks=True,
    )
    masks3d = result[0] if isinstance(result, (list, tuple)) else result

    mask_path = out_root / "masks" / f"{path.stem}_masks3d.tif"
    tiff.imwrite(mask_path.as_posix(), masks3d.astype(np.int32))

    qc_path = out_root / "overlays" / f"{path.stem}_mip_composite.png"
    save_qc_mips(stack, masks3d, qc_path)
    print(f"Done. Masks: {mask_path}\nQC composite: {qc_path}")


if __name__ == "__main__":
    main()
