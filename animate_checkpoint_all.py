#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Animate predictions from a trained checkpoint for a list of variables.

Example:
  python animate_checkpoint_all.py \
    --root /storage/home/sbr5878/scratch/new_diffusion/MERRA2_splits \
    --split val \
    --config refined_config.yaml \
    --ckpt /storage/home/sbr5878/ISCL/rishi/diffusion/Merra2/checkpoints/merra2_nextstep_best.pt \
    --outdir /storage/home/sbr5878/scratch/new_diffusion/MERRA2_gifs \
    --vars temperature_300 temperature_925 \
    --start 0 --nframes 24 --stride 1 \
    --physical --with_diff
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
import yaml

from merra2_nc_dataloader import MERRA2SR
from model import ForecastModel


# ----------------- small utils -----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H,W,4)
    return buf[..., :3].copy()

def robust_range(arrs, pct=(1, 99)):
    arrs = np.asarray(arrs)
    lo = np.nanpercentile(arrs, pct[0])
    hi = np.nanpercentile(arrs, pct[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(arrs)), float(np.nanmax(arrs))
    return lo, hi

def center_crop_hw_t(x: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    if H == H_out and W == W_out:
        return x
    top  = max((H - H_out) // 2, 0)
    left = max((W - W_out) // 2, 0)
    return x[..., top:top+H_out, left:left+W_out]

def flatten_semicolon_list(lst):
    """Support ['a; b; c', 'd'] style into flat token list."""
    out = []
    for s in lst:
        if isinstance(s, str):
            out += [t.strip() for t in s.split(";") if t.strip()]
        else:
            out.append(str(s))
    return out

def to_numpy_ch(x):
    """(C,H,W) torch or np -> np"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# ----------------- GIF builder -----------------
def make_one_var_gif(
    ds: MERRA2SR,
    var_name: str,
    var_index: int,
    model: torch.nn.Module,
    device: torch.device,
    img_hw: Tuple[int,int],
    start: int,
    nframes: int,
    stride: int,
    outdir: str,
    split: str,
    fps: int = 6,
    physical: bool = False,
    with_diff: bool = False,
):
    """
    Columns:
      [ xb(t) | tb(t+1) | model_pred(t→t+1) | (optional) |GT - pred| ]
    """
    Hc, Wc = img_hw
    lat, lon = ds.get_lat_lon()
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # means/stds from ds (np arrays [C,1,1])
    means = ds.means
    stds  = ds.stds

    # ---- probe frames for robust color limits ----
    probe_idx = [start + i * stride for i in range(min(12, nframes))]
    xb_stack, tb_stack = [], []
    for k in probe_idx:
        xb, tb = ds[k]         # tensors [C,H,W]
        xb_np = to_numpy_ch(xb)
        tb_np = to_numpy_ch(tb)
        xb1 = xb_np[var_index]
        tb1 = tb_np[var_index]
        if physical:
            m = float(means[var_index, 0, 0])
            s = float(stds[var_index, 0, 0])
            xb1 = xb1 * s + m
            tb1 = tb1 * s + m
        xb_stack.append(xb1)
        tb_stack.append(tb1)

    xb_vmin, xb_vmax = robust_range(xb_stack)
    tb_vmin, tb_vmax = robust_range(tb_stack)

    # ---- frames ----
    frames = []
    for i in range(nframes):
        k = start + i * stride
        xb, tb = ds[k]                     # [C,H,W] tensors
        t0, t1 = ds.get_time(k)

        # model prediction
        xbt = xb.unsqueeze(0).to(device)   # [1,C,H,W]
        xbt_c = center_crop_hw_t(xbt, Hc, Wc)
        with torch.no_grad():
            ybt_c = model(xbt_c)           # [1,Cout,Hc,Wc]

        # resize back to original grid for visualization
        H0, W0 = xb.shape[-2:]
        if (Hc, Wc) != (H0, W0):
            ybt = torch.nn.functional.interpolate(ybt_c, size=(H0, W0), mode="nearest")
        else:
            ybt = ybt_c

        xb_np = to_numpy_ch(xb)            # (C,H,W)
        tb_np = to_numpy_ch(tb)
        yp_np = ybt[0].detach().cpu().numpy()

        xb1 = xb_np[var_index]
        tb1 = tb_np[var_index]
        yp1 = yp_np[var_index]

        if physical:
            m = float(means[var_index, 0, 0])
            s = float(stds[var_index, 0, 0])
            xb1 = xb1 * s + m
            tb1 = tb1 * s + m
            yp1 = yp1 * s + m

        ncols = 4 if with_diff else 3
        fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 3.6), dpi=120)

        im0 = axs[0].imshow(xb1, origin="lower", extent=extent, aspect="auto",
                            vmin=xb_vmin, vmax=xb_vmax, cmap="viridis")
        axs[0].set_title(f"{var_name}\ninput @ t")
        axs[0].set_xlabel("Lon"); axs[0].set_ylabel("Lat")
        fig.colorbar(im0, ax=axs[0], shrink=0.8)

        im1 = axs[1].imshow(tb1, origin="lower", extent=extent, aspect="auto",
                            vmin=tb_vmin, vmax=tb_vmax, cmap="viridis")
        axs[1].set_title(f"{var_name}\nGT @ t+1")
        axs[1].set_xlabel("Lon"); axs[1].set_ylabel("Lat")
        fig.colorbar(im1, ax=axs[1], shrink=0.8)

        im2 = axs[2].imshow(yp1, origin="lower", extent=extent, aspect="auto",
                            vmin=tb_vmin, vmax=tb_vmax, cmap="viridis")
        axs[2].set_title(f"{var_name}\nModel pred (t→t+1)")
        axs[2].set_xlabel("Lon"); axs[2].set_ylabel("Lat")
        fig.colorbar(im2, ax=axs[2], shrink=0.8)

        if with_diff:
            diff = np.abs(tb1 - yp1)
            dmin, dmax = robust_range(diff[None, ...], pct=(1, 99))
            im3 = axs[3].imshow(diff, origin="lower", extent=extent, aspect="auto",
                                vmin=dmin, vmax=dmax, cmap="magma")
            axs[3].set_title("|GT - pred|")
            axs[3].set_xlabel("Lon"); axs[3].set_ylabel("Lat")
            fig.colorbar(im3, ax=axs[3], shrink=0.8)

        fig.suptitle(
            f"{split}  t={np.datetime_as_string(t0, unit='m')}  →  t+1={np.datetime_as_string(t1, unit='m')}",
            fontsize=10, y=0.98
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    outname = f"{split}_{var_name}_t{start}_n{nframes}_s{stride}_pred.gif"
    ensure_dir(outdir)
    outpath = os.path.join(outdir, outname)
    imageio.mimsave(outpath, frames, fps=fps, loop=0)
    print("Saved:", outpath)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vars", nargs="*", default=None,
                    help="tokens to animate; default = targets from training config")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--outdir", default="./gifs")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--physical", action="store_true")
    ap.add_argument("--with_diff", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # ---- load YAML for img_resolution fallback ----
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f) or {}
    Hc, Wc = tuple(yaml_cfg.get("model", {}).get("img_resolution", [361, 576]))

    # ---- load checkpoint (prefer cfg inside) ----
    chk = torch.load(args.ckpt, map_location="cpu")
    internal_cfg = chk.get("cfg", None)

    if internal_cfg is not None:
        data_cfg = internal_cfg["data"]
        inputs = data_cfg["inputs"]
        targets = data_cfg["targets"]
    else:
        # fallback to YAML-only
        d = yaml_cfg.get("data", {})
        inputs  = d.get("inputs") or d.get("variables")
        targets = d.get("targets") or inputs
        if inputs is None or targets is None:
            raise ValueError("Need data.inputs/targets or data.variables in config.")
    inputs  = flatten_semicolon_list(inputs)
    targets = flatten_semicolon_list(targets)

    # ---- build model ----
    in_ch  = len(inputs)
    out_ch = len(targets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ForecastModel(
        in_channels=in_ch,
        out_channels=out_ch,
        img_resolution=(Hc, Wc),
        dim=yaml_cfg.get("model", {}).get("dim", 256),
        depth=yaml_cfg.get("model", {}).get("depth", 8),
        heads=yaml_cfg.get("model", {}).get("heads", 8),
        patch_size=tuple(yaml_cfg.get("model", {}).get("patch_size", [1, 2])),
        window_size=tuple(yaml_cfg.get("model", {}).get("window_size", [4, 8])),
        flash=yaml_cfg.get("model", {}).get("flash", True),
    ).to(device)

    state = chk.get("model", chk)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded model from {args.ckpt}")
    print(f"in_ch={in_ch}, out_ch={out_ch}")

    # ---- dataset with the SAME channel order as training ----
    ds = MERRA2SR(root=args.root, variables=inputs, split=args.split)

    # ---- which vars to animate ----
    if args.vars and len(args.vars) > 0:
        animate_tokens = args.vars
    else:
        animate_tokens = targets  # show predicted channels by default

    idx = {tok: i for i, tok in enumerate(targets)}
    for tok in animate_tokens:
        if tok not in idx:
            raise ValueError(f"Requested variable '{tok}' not in targets list from training.")
    # we assume inputs == targets order (true for your 52-var run)

    # ---- go ----
    for tok in animate_tokens:
        print(f"[INFO] Rendering: {tok}")
        vi = idx[tok]
        make_one_var_gif(
            ds=ds,
            var_name=tok,
            var_index=vi,
            model=model,
            device=device,
            img_hw=(Hc, Wc),
            start=args.start,
            nframes=args.nframes,
            stride=args.stride,
            outdir=args.outdir,
            split=args.split,
            fps=args.fps,
            physical=args.physical,
            with_diff=args.with_diff,
        )

if __name__ == "__main__":
    main()
