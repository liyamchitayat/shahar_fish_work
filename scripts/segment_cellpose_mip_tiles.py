#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import numpy as np
import tifffile as tiff

# Use local caches for models and MPL to avoid permission issues
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(Path.cwd() / "cellpose_models"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import nd2
from cellpose import models
from cellpose import plot as cp_plot


def zmax_projection_for_channel(nd2_path: str, channel: int) -> np.ndarray:
    with nd2.ND2File(nd2_path) as f:
        frame_count = getattr(f, "_frame_count", None)
        if frame_count is None:
            sizes = dict(f.sizes)
            frame_count = sizes.get('Z', 1)
        mip = None
        for z in range(frame_count):
            fr = f.read_frame(z)  # (C, Y, X)
            if channel < 0 or channel >= fr.shape[0]:
                raise ValueError(f"Channel {channel} out of range (C={fr.shape[0]}) for {nd2_path}")
            plane = fr[channel]
            mip = plane.astype(np.uint16) if mip is None else np.maximum(mip, plane)
        return mip


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max() or 1.0)
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return img


def save_composite(image: np.ndarray, masks: np.ndarray, out_png: Path):
    img = normalize_for_display(image)
    orig_rgb = np.dstack([img, img, img])
    mask_rgb = cp_plot.mask_rgb(masks)
    overlay = cp_plot.mask_overlay(img, masks)
    def to_float01(a):
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        return np.clip(a, 0, 1)
    comp = np.concatenate([to_float01(orig_rgb), to_float01(mask_rgb), to_float01(overlay)], axis=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib import pyplot as plt
    plt.imsave(out_png.as_posix(), comp)


def run_cellpose(image: np.ndarray, diameter: float, cellprob: float, flowth: float):
    model = models.CellposeModel(gpu=False)
    result = model.eval(
        image,
        channels=[0, 0],
        diameter=diameter,
        cellprob_threshold=cellprob,
        flow_threshold=flowth,
        normalize=True,
        augment=False,
        compute_masks=True,
        tile_overlap=0.1,
    )
    if isinstance(result, (list, tuple)):
        masks = result[0]
    else:
        masks = result
    return masks


def main():
    ap = argparse.ArgumentParser(description="Run Cellpose on two tiles of the MIP (Z-max) without modifying the main script.")
    ap.add_argument("nd2_path", help="ND2 file path")
    ap.add_argument("--channel", type=int, required=True, help="Channel index for MIP")
    ap.add_argument("--tile", type=int, default=1024, help="Tile size (pixels)")
    ap.add_argument("--outdir", default="segmentation", help="Output directory root")
    ap.add_argument("--diameter", type=float, default=60.0, help="Cell diameter in pixels")
    ap.add_argument("--cellprob", type=float, default=-5.0, help="Very permissive cell prob threshold")
    ap.add_argument("--flowth", type=float, default=0.0, help="Very permissive flow threshold")
    args = ap.parse_args()

    nd2_path = Path(args.nd2_path)
    base = nd2_path.stem
    out_root = Path(args.outdir) / base
    (out_root / "projections").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "overlays").mkdir(parents=True, exist_ok=True)

    print(f"Computing Z-max MIP for {nd2_path} ch {args.channel} ...", flush=True)
    mip = zmax_projection_for_channel(nd2_path.as_posix(), args.channel)
    H, W = mip.shape
    t = min(args.tile, H, W)

    tiles = [
        (0, 0),  # top-left
        (max(0, H - t), max(0, W - t)),  # bottom-right
    ]

    # Save full MIP for reference
    mip_path = out_root / "projections" / f"{base}_ch{args.channel}_zmax_full.tif"
    if not mip_path.exists():
        tiff.imwrite(mip_path.as_posix(), mip, dtype=mip.dtype)

    for (y0, x0) in tiles:
        tile = mip[y0:y0+t, x0:x0+t]
        tag = f"tile_y{y0}_x{x0}_h{t}_w{t}"
        print(f"Segmenting {tag} ...", flush=True)
        masks = run_cellpose(tile, diameter=args.diameter, cellprob=args.cellprob, flowth=args.flowth)

        mask_path = out_root / "masks" / f"{base}_ch{args.channel}_zmax_{tag}.tif"
        tiff.imwrite(mask_path.as_posix(), masks.astype(np.int32))

        composite_path = out_root / "overlays" / f"{base}_ch{args.channel}_zmax_composite_{tag}.png"
        save_composite(tile, masks, composite_path)
        print(f"Saved: {mask_path.name}, {composite_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()

