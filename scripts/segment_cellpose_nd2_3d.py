#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import tifffile as tiff

# Ensure local caches are writable
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(Path.cwd() / "cellpose_models"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import nd2
from cellpose import models
from cellpose import plot as cp_plot
import matplotlib.pyplot as plt


def read_xy_z_calibration(nd2_path: str, ch_index: int = 0):
    with nd2.ND2File(nd2_path) as f:
        md = f.frame_metadata(0)
        # find matching channel entry
        vol = None
        if hasattr(md, 'channels') and md.channels:
            chs = md.channels
            for ch in chs:
                if getattr(getattr(ch, 'channel', None), 'index', None) == ch_index:
                    vol = getattr(ch, 'volume', None)
                    break
            if vol is None:
                vol = getattr(chs[0], 'volume', None)
        if vol and getattr(vol, 'axesCalibration', None):
            xy_um, xy_um_y, z_um = vol.axesCalibration
            # some files store (X,Y,Z); assume X==Y
            return float(xy_um), float(z_um)
    # Fallbacks
    return 0.1696, 5.0


def read_z_stack(nd2_path: str, ch: int, z0: int, z1: int) -> np.ndarray:
    with nd2.ND2File(nd2_path) as f:
        sizes = dict(f.sizes)
        Z = sizes.get('Z', 1)
        yx = None
        z0c = max(0, z0)
        z1c = min(Z, z1)
        stack = None
        for z in range(z0c, z1c):
            fr = f.read_frame(z)  # (C, Y, X)
            if ch < 0 or ch >= fr.shape[0]:
                raise ValueError(f"Channel {ch} out of range for file (C={fr.shape[0]})")
            plane = fr[ch]
            if stack is None:
                H, W = plane.shape
                stack = np.empty((z1c - z0c, H, W), dtype=plane.dtype)
            stack[z - z0c] = plane
        return stack


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max() or 1.0)
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return img


def save_qc_mips(stack: np.ndarray, masks3d: np.ndarray, out_png: Path):
    # XY MIP for image and masks
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
    ap = argparse.ArgumentParser(description="3D Cellpose segmentation on ND2 Z-substack using original 2D params (MIP workflow left untouched).")
    ap.add_argument("nd2_path", help="ND2 path")
    ap.add_argument("--channel", type=int, required=True, help="Channel index")
    ap.add_argument("--z0", type=int, default=0, help="Start Z (inclusive)")
    ap.add_argument("--z1", type=int, default=10, help="End Z (exclusive)")
    ap.add_argument("--outdir", default="segmentation3d", help="Output root directory")
    # original parameters
    ap.add_argument("--diameter", type=float, default=60.0, help="XY diameter (pixels)")
    ap.add_argument("--cellprob", type=float, default=0.0, help="Cell probability threshold")
    ap.add_argument("--flowth", type=float, default=0.4, help="Flow threshold")
    ap.add_argument("--anisotropy", type=float, default=None, help="Optional override for anisotropy (Z/XY) if None infer from ND2 metadata")
    ap.add_argument("--crop", type=int, default=None, help="Optional center crop (pixels) to reduce compute")
    args = ap.parse_args()

    nd2p = Path(args.nd2_path)
    base = nd2p.stem
    out_base = Path(args.outdir) / base
    (out_base / "masks").mkdir(parents=True, exist_ok=True)
    (out_base / "projections").mkdir(parents=True, exist_ok=True)
    (out_base / "overlays").mkdir(parents=True, exist_ok=True)

    xy_um, z_um = read_xy_z_calibration(nd2p.as_posix(), args.channel)
    anis = args.anisotropy if args.anisotropy is not None else (z_um / max(xy_um, 1e-6))
    print(f"XY pixel = {xy_um:.4f} µm, Z step = {z_um:.4f} µm, anisotropy ~ {anis:.2f}")

    stack = read_z_stack(nd2p.as_posix(), args.channel, args.z0, args.z1)
    if args.crop is not None:
        H, W = stack.shape[1:]
        s = min(args.crop, H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        stack = stack[:, y0:y0+s, x0:x0+s]

    # Save substack projection for reference
    mip2d = stack.max(axis=0)
    mip_path = out_base / "projections" / f"{base}_ch{args.channel}_z{args.z0}-{args.z1}_zmax_3dref.tif"
    tiff.imwrite(mip_path.as_posix(), mip2d, dtype=stack.dtype)

    # Run Cellpose 3D with original params
    model = models.CellposeModel(gpu=False)
    result = model.eval(
        stack,
        do_3D=True,
        z_axis=0,
        diameter=args.diameter,
        cellprob_threshold=args.cellprob,
        flow_threshold=args.flowth,
        anisotropy=anis,
        normalize=True,
        augment=False,
        compute_masks=True,
    )
    masks3d = result[0] if isinstance(result, (list, tuple)) else result

    mask_path = out_base / "masks" / f"{base}_ch{args.channel}_z{args.z0}-{args.z1}_masks3d.tif"
    tiff.imwrite(mask_path.as_posix(), masks3d.astype(np.int32))

    # QC composite from MIPs
    comp_path = out_base / "overlays" / f"{base}_ch{args.channel}_z{args.z0}-{args.z1}_mip_composite.png"
    save_qc_mips(stack, masks3d, comp_path)

    print(f"Done. 3D masks: {mask_path}\nQC composite (MIPs): {comp_path}")


if __name__ == "__main__":
    main()

