#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np

# Ensure model cache + matplotlib cache are writable inside workspace before importing cellpose
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(Path.cwd() / "cellpose_models"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import nd2
from cellpose import models
from cellpose import plot as cp_plot
import tifffile as tiff
import matplotlib.pyplot as plt


def zmax_projection_for_channel(nd2_path: str, channel: int) -> np.ndarray:
    """Stream through Z frames and compute a max-intensity projection for a given channel.

    Returns a 2D numpy uint16 image (Y, X).
    """
    with nd2.ND2File(nd2_path) as f:
        frame_count = getattr(f, "_frame_count", None)
        if frame_count is None:
            # fallback to Z size
            sizes = dict(f.sizes)
            frame_count = sizes.get('Z', 1)

        mip = None
        for z in range(frame_count):
            fr = f.read_frame(z)  # shape: (C, Y, X) for this Z
            if channel < 0 or channel >= fr.shape[0]:
                raise ValueError(f"Channel index {channel} out of range for file {nd2_path} (C={fr.shape[0]})")
            plane = fr[channel]  # (Y, X), dtype likely uint16
            if mip is None:
                mip = plane.astype(np.uint16, copy=True)
            else:
                np.maximum(mip, plane, out=mip)
        return mip


def save_overlay(image: np.ndarray, masks: np.ndarray, out_png: Path):
    """Save an RGB overlay of masks outlines over a contrast-scaled grayscale image."""
    img = normalize_for_display(image)
    # Use v4 API: direct mask overlay
    overlay = cp_plot.mask_overlay(img, masks)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png.as_posix(), overlay)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max() or 1.0)
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    return img


def save_composite(image: np.ndarray, masks: np.ndarray, out_png: Path):
    """Save a composite PNG with 3 panels: original MIP, masks-only, and overlay."""
    img = normalize_for_display(image)
    # original as RGB
    orig_rgb = np.dstack([img, img, img])
    # masks-only color label map
    mask_rgb = cp_plot.mask_rgb(masks)
    # overlay
    overlay = cp_plot.mask_overlay(img, masks)
    # ensure all float in [0,1]
    def to_float01(a):
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        return np.clip(a, 0, 1)
    comp = np.concatenate([to_float01(orig_rgb), to_float01(mask_rgb), to_float01(overlay)], axis=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_png.as_posix(), comp)


def run_cellpose(
    image: np.ndarray,
    model_type: str = "cyto2",
    diameter: float = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    rescale: float = 1.0,
):
    # Instantiate Cellpose v4 model (defaults to cpsam). Use CPU unless user has GPU/MPS and Cellpose auto-detects.
    model = models.CellposeModel(gpu=False)

    # channels deprecated in v4, but still supported for up to 3 chans; pass None to let it infer single-channel
    result = model.eval(
        image,
        channels=[0, 0],  # grayscale single channel
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        rescale=rescale,
        normalize=True,
        augment=False,
        compute_masks=True,
        tile_overlap=0.1,
    )
    if isinstance(result, (list, tuple)) and len(result) == 4:
        masks, flows, styles, diams = result
    elif isinstance(result, (list, tuple)) and len(result) == 3:
        masks, flows, styles = result
        diams = None
    else:
        # Unexpected
        masks = result
        flows, styles, diams = None, None, None
    return masks, flows, styles, diams


def main():
    ap = argparse.ArgumentParser(description="Segment ND2 images with Cellpose on a Z-max projection of a specified channel.")
    ap.add_argument("inputs", nargs="+", help="One or more ND2 files")
    ap.add_argument("--channels", nargs="+", type=int, required=True, help="Channel index for each input ND2 (same order).")
    ap.add_argument("--outdir", default="segmentation", help="Output directory root.")
    ap.add_argument("--model", default="cyto2", choices=["cyto", "cyto2", "nuclei"], help="(Ignored in v4) kept for CLI compat.")
    ap.add_argument("--diameter", type=float, default=None, help="Cell diameter estimate (pixels). Let Cellpose auto-estimate if omitted.")
    ap.add_argument("--cellprob", type=float, default=0.0, help="Cell probability threshold.")
    ap.add_argument("--flowth", type=float, default=0.4, help="Flow threshold.")
    ap.add_argument("--rescale", type=float, default=1.0, help="Image rescale factor before segmentation.")
    ap.add_argument("--crop", type=int, default=None, help="Optional center crop size (pixels), e.g., 1024 for quick test.")
    ap.add_argument("--composite-only", action="store_true", help="Do not run Cellpose; load saved MIP + masks and make composite.")
    ap.add_argument("--force", action="store_true", help="Force rerun even if outputs exist.")
    args = ap.parse_args()

    if len(args.inputs) != len(args.channels):
        raise SystemExit("Number of --channels must match number of input files")

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    for nd2_path, ch in zip(args.inputs, args.channels):
        nd2_path = Path(nd2_path)
        print(f"Processing {nd2_path} (channel {ch}) ...", flush=True)
        try:
            mip = zmax_projection_for_channel(nd2_path.as_posix(), ch)
        except Exception as e:
            print(f"ERROR computing Z-projection for {nd2_path.name}: {e}")
            continue

        # Compute output suffix to avoid overwriting
        base = nd2_path.stem
        out_dir = outroot / base
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        (out_dir / "projections").mkdir(parents=True, exist_ok=True)

        suffix = f"_crop{args.crop}" if args.crop is not None else "_full"
        mip_path = out_dir / "projections" / f"{base}_ch{ch}_zmax{suffix}.tif"
        mask_path = out_dir / "masks" / f"{base}_ch{ch}_zmax_masks{suffix}.tif"
        overlay_path = out_dir / "overlays" / f"{base}_ch{ch}_zmax_overlay{suffix}.png"
        composite_path = out_dir / "overlays" / f"{base}_ch{ch}_zmax_composite{suffix}.png"

        if args.composite_only:
            # Load saved mip + masks and make composite only
            if mip_path.exists() and mask_path.exists():
                mip = tiff.imread(mip_path.as_posix())
                masks = tiff.imread(mask_path.as_posix())
                try:
                    save_composite(mip, masks, composite_path)
                except Exception as e:
                    print(f"Warning: failed to create composite for {base}: {e}")
                print(f"Composite saved: {composite_path}")
                continue
            else:
                print(f"Composite-only requested but missing files: {mip_path} or {mask_path}")
                continue

        # Optional center crop for quick testing
        if args.crop is not None:
            h, w = mip.shape
            s = min(args.crop, h, w)
            y0 = (h - s) // 2
            x0 = (w - s) // 2
            mip = mip[y0:y0+s, x0:x0+s]

        # Save the (possibly cropped) MIP for reference
        if not mip_path.exists() or args.force:
            tiff.imwrite(mip_path.as_posix(), mip, dtype=mip.dtype)

        # Skip model if masks already exist and not forcing
        if mask_path.exists() and not args.force:
            print(f"Masks exist, skipping model: {mask_path}")
            masks = tiff.imread(mask_path.as_posix())
        else:
            # Run Cellpose
            print(f"Running Cellpose on {base} ...", flush=True)
            masks, flows, styles, diams = run_cellpose(
                mip,
                model_type=args.model,
                diameter=args.diameter,
                cellprob_threshold=args.cellprob,
                flow_threshold=args.flowth,
                rescale=args.rescale,
            )
            # Save mask as TIFF
            tiff.imwrite(mask_path.as_posix(), masks.astype(np.int32))

            # Save Cellpose npy outputs (flows etc.) for reproducibility
            npy_dir = out_dir / "npy"
            npy_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_dir / f"{base}_ch{ch}_zmax_masks{suffix}.npy", masks)
            try:
                np.save(npy_dir / f"{base}_ch{ch}_zmax_flows{suffix}.npy", np.array(flows, dtype=object))
            except Exception:
                pass
            if diams is not None:
                np.save(npy_dir / f"{base}_ch{ch}_zmax_diams{suffix}.npy", np.array(diams))

        # Save overlay PNG
        try:
            save_overlay(mip, masks, overlay_path)
        except Exception as e:
            print(f"Warning: failed to create overlay for {base}: {e}")

        # Save composite PNG (orig | masks | overlay)
        try:
            save_composite(mip, masks, composite_path)
        except Exception as e:
            print(f"Warning: failed to create composite for {base}: {e}")

        print(f"Done: {base}. Masks: {mask_path}")


if __name__ == "__main__":
    main()
