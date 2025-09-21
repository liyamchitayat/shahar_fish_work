#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
from skimage.measure import regionprops
from math import pi
import os

# Use local caches for MPL to avoid permission issues when saving PNGs
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

from cellpose import plot as cp_plot
import matplotlib.pyplot as plt


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max() or 1.0)
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return img


def filter_by_roundness_and_size(mask: np.ndarray, min_d: float, max_d: float, min_circ: float,
                                 min_area: int = 0, max_area: int = None) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int32)
    lbl = 1
    # regionprops expects labels starting at 1
    for r in regionprops(mask):
        area = r.area
        if area <= 0:
            continue
        if min_area and area < min_area:
            continue
        if max_area and area > max_area:
            continue
        # Equivalent diameter
        d_eq = r.equivalent_diameter
        if (min_d and d_eq < min_d) or (max_d and d_eq > max_d):
            continue
        # Circularity = 4*pi*Area / Perimeter^2
        perim = r.perimeter
        if perim <= 0:
            continue
        circ = 4.0 * pi * area / (perim * perim)
        if circ < min_circ:
            continue
        out[mask == r.label] = lbl
        lbl += 1
    return out


def save_overlay_and_composite(image: np.ndarray, masks: np.ndarray, out_overlay: Path, out_composite: Path):
    img = normalize_for_display(image)
    overlay = cp_plot.mask_overlay(img, masks)
    mask_rgb = cp_plot.mask_rgb(masks)
    orig_rgb = np.dstack([img, img, img])
    def to_float01(a):
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        return np.clip(a, 0, 1)
    comp = np.concatenate([to_float01(orig_rgb), to_float01(mask_rgb), to_float01(overlay)], axis=1)
    out_overlay.parent.mkdir(parents=True, exist_ok=True)
    out_composite.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_overlay.as_posix(), overlay)
    plt.imsave(out_composite.as_posix(), comp)


def main():
    ap = argparse.ArgumentParser(description="Filter Cellpose masks by roundness and diameter range (MIP-only).")
    ap.add_argument("--mask", required=True, help="Path to masks TIFF from MIP (_zmax_*.tif)")
    ap.add_argument("--mip", required=True, help="Path to original MIP TIFF used for segmentation")
    ap.add_argument("--outdir", default=None, help="Output directory; defaults next to mask")
    ap.add_argument("--min-diam", type=float, default=40.0, help="Min equivalent diameter (pixels)")
    ap.add_argument("--max-diam", type=float, default=90.0, help="Max equivalent diameter (pixels)")
    ap.add_argument("--min-circularity", type=float, default=0.6, help="Min circularity (4πA/P²)")
    ap.add_argument("--min-area", type=int, default=0, help="Optional min area in pixels")
    ap.add_argument("--max-area", type=int, default=0, help="Optional max area in pixels; 0 disables")
    args = ap.parse_args()

    mask_path = Path(args.mask)
    mip_path = Path(args.mip)
    outdir = Path(args.outdir) if args.outdir else mask_path.parent

    mask = tiff.imread(mask_path.as_posix())
    mip = tiff.imread(mip_path.as_posix())

    # If dimensions mismatch and mask filename encodes tile coords, crop MIP to tile
    if mip.shape != mask.shape:
        import re
        m = re.search(r"tile_y(\d+)_x(\d+)_h(\d+)_w(\d+)", mask_path.stem)
        if m:
            y0, x0, h, w = map(int, m.groups())
            mip = mip[y0:y0+h, x0:x0+w]
        else:
            raise SystemExit("MIP and mask dimensions differ and tile coordinates not found in filename.")

    max_area = args.max_area if args.max_area and args.max_area > 0 else None
    filt = filter_by_roundness_and_size(
        mask,
        min_d=args.min_diam,
        max_d=args.max_diam,
        min_circ=args.min_circularity,
        min_area=args.min_area,
        max_area=max_area,
    )

    # Save filtered mask and overlays
    stem = mask_path.stem
    filt_mask_path = outdir / f"{stem}_filtered.tif"
    tiff.imwrite(filt_mask_path.as_posix(), filt.astype(np.int32))

    overlay_path = outdir.parent / "overlays" / f"{stem}_filtered_overlay.png"
    composite_path = outdir.parent / "overlays" / f"{stem}_filtered_composite.png"
    save_overlay_and_composite(mip, filt, overlay_path, composite_path)

    # Simple stats
    vals = np.unique(filt)
    num = int((vals > 0).sum())
    print(f"Filtered cells kept: {num}\nSaved: {filt_mask_path}\nOverlay: {overlay_path}\nComposite: {composite_path}")


if __name__ == "__main__":
    main()
