
import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from merra2_nc_dataloader import MERRA2SR
from model import ForecastModel


# ----------------- small helpers -----------------

def flatten_semicolon_list(lst):
    """Supports ['a; b; c', 'd'] style lists from YAML."""
    out = []
    for s in lst:
        if isinstance(s, str) and ";" in s:
            out += [t.strip() for t in s.split(";") if t.strip()]
        else:
            out.append(str(s))
    return out


def center_crop_indices(H: int, W: int, Hc: int, Wc: int) -> Tuple[slice, slice]:
    top  = max((H - Hc) // 2, 0)
    left = max((W - Wc) // 2, 0)
    return slice(top, top + Hc), slice(left, left + Wc)


def center_crop_hw_torch(x: torch.Tensor, Hc: int, Wc: int) -> torch.Tensor:
    # x: [C,H,W] or [B,C,H,W]
    H, W = x.shape[-2:]
    if (H, W) == (Hc, Wc):
        return x
    sH, sW = center_crop_indices(H, W, Hc, Wc)
    return x[..., sH, sW]


def compute_rmse(pred: np.ndarray, gt: np.ndarray, axis=None):
    """NaN-safe RMSE."""
    diff2 = (pred - gt) ** 2
    return np.sqrt(np.nanmean(diff2, axis=axis))


# ----------------- main -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                  help="MERRA2_splits root (same as training)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--config", required=True,
                  help="YAML config used for training (e.g. refined_config.yaml)")
    p.add_argument("--ckpt", required=True,
                  help="Checkpoint path (e.g. ./checkpoints/merra2_nextstep_best.pt)")
    p.add_argument("--start_index", type=int, required=True,
                  help="Dataset index to start from (t0)")
    p.add_argument("--nsteps", type=int, default=56,
                  help="Number of forecast steps (3h per step for M2I3NPASM)")
    p.add_argument("--variables", nargs="*", default=None,
                  help=("Subset of tokens to plot RMSE for (must be in training variables, order matters). "
                        "If omitted, uses ALL target variables from config."))
    p.add_argument("--outdir", required=True,
                  help="Directory to save CSV and plots")
    p.add_argument("--tag", default="rmse_from_ckpt_autoreg",
                  help="Tag prefix for output files")
    p.add_argument("--save_individual", action="store_true",
                  help="Also write per-variable PNGs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- load config ----------
    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    data_cfg = cfg_yaml.get("data", {})
    model_cfg = cfg_yaml.get("model", {})

    inputs  = data_cfg.get("inputs")
    targets = data_cfg.get("targets")
    variables = data_cfg.get("variables")

    if inputs is None and variables is not None:
        inputs = variables
    if targets is None and variables is not None:
        targets = variables
    if inputs is None or targets is None:
        raise ValueError("Config must define data.inputs/data.targets, or data.variables.")

    inputs  = flatten_semicolon_list(inputs)
    targets = flatten_semicolon_list(targets)

    # You trained with in_channels=len(inputs), out_channels=len(targets),
    # and in your setup inputs == targets == 52-variable list.
    train_tokens = targets

    # Which variables to plot RMSE for?
    if args.variables and len(args.variables) > 0:
        plot_tokens = list(args.variables)
        for v in plot_tokens:
            if v not in train_tokens:
                raise ValueError(f"Requested variable '{v}' not in training tokens {train_tokens}")
    else:
        plot_tokens = train_tokens

    idx_map: Dict[str, int] = {tok: i for i, tok in enumerate(train_tokens)}
    plot_indices = [idx_map[v] for v in plot_tokens]
    C_plot = len(plot_tokens)

    # Model spatial resolution
    Hc, Wc = tuple(model_cfg.get("img_resolution", [361, 576]))

    # ---------- dataset ----------
    ds = MERRA2SR(root=args.root, variables=train_tokens, split=args.split)
    means = torch.from_numpy(ds.means.squeeze(-1).squeeze(-1))  # [C]
    stds  = torch.from_numpy(ds.stds.squeeze(-1).squeeze(-1))   # [C]

    # Check we have enough timesteps for requested horizon
    if args.start_index + args.nsteps >= len(ds):
        raise ValueError(
            f"Not enough samples: start_index={args.start_index}, nsteps={args.nsteps}, len(ds)={len(ds)}"
        )

    # ---------- model ----------
    in_ch  = len(train_tokens)
    out_ch = len(train_tokens)

    model = ForecastModel(
        in_channels=in_ch,
        out_channels=out_ch,
        img_resolution=(Hc, Wc),
        dim=model_cfg.get("dim", 256),
        depth=model_cfg.get("depth", 8),
        heads=model_cfg.get("heads", 8),
        patch_size=tuple(model_cfg.get("patch_size", [1, 2])),
        window_size=tuple(model_cfg.get("window_size", [4, 8])),
        flash=model_cfg.get("flash", True),
    ).to(device)

    chk = torch.load(args.ckpt, map_location="cpu")
    state = chk.get("model", chk)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------- autoregressive rollout ----------
    T = args.nsteps
    rmse_t_c = np.zeros((T, C_plot), dtype=np.float64)
    rmse_overall = np.zeros(T, dtype=np.float64)

    # seed: xb at t0
    xb_std_seed, tb_std_seed = ds[args.start_index]  # [C,H,W] each
    xb_curr_std = center_crop_hw_torch(xb_std_seed, Hc, Wc).to(device)  # [C,Hc,Wc]

    with torch.no_grad():
        for s in range(T):
            # 1) predict x_hat at time t0 + (s+1)
            pred_std = model(xb_curr_std.unsqueeze(0)).squeeze(0).cpu()  # [C,Hc,Wc]

            # 2) ground truth for that horizon:
            #    for step s (0-based), we want x(t0 + s + 1) in standardized space,
            #    which is tb_std from sample index (start_index + s)
            _, tb_std_gt = ds[args.start_index + s]  # [C,H,W]
            tb_std_gt = center_crop_hw_torch(tb_std_gt, Hc, Wc)

            # 3) destandardize to physical units for ALL channels
            pred_phys = pred_std.clone()
            gt_phys   = tb_std_gt.clone()
            for c in range(in_ch):
                m = means[c]
                s_ = stds[c]
                pred_phys[c] = pred_std[c] * s_ + m
                gt_phys[c]   = gt_phys[c] * s_ + m

            pred_np = pred_phys.numpy()
            gt_np   = gt_phys.numpy()

            # 4) per-variable RMSE (only for the subset we care about)
            for j, c_idx in enumerate(plot_indices):
                rmse_t_c[s, j] = compute_rmse(pred_np[c_idx], gt_np[c_idx], axis=(-2, -1))

            # overall RMSE across all channels/pixels
            mse_all = ((pred_np - gt_np) ** 2).mean()
            rmse_overall[s] = np.sqrt(mse_all)

            # 5) update xb_curr_std for autoregressive step
            xb_curr_std = pred_std.to(device)

    # ---------- save CSV ----------
    steps = np.arange(1, T + 1)
    hours = steps * 3
    days  = hours / 24.0

    base = args.tag
    csv_path = os.path.join(args.outdir, f"{base}_rmse.csv")

    header = ["step_1based", "hours", "days"] \
             + [f"rmse_{v}" for v in plot_tokens] \
             + ["rmse_overall"]

    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for t in range(T):
            row = [steps[t], hours[t], days[t]] \
                  + [rmse_t_c[t, j] for j in range(C_plot)] \
                  + [rmse_overall[t]]
            f.write(",".join(
                f"{x}" if isinstance(x, int) else f"{x:.6f}"
                for x in row
            ) + "\n")

    print(f"[INFO] Saved RMSE CSV -> {csv_path}")

    # ---------- plot RMSE vs time ----------
    ncols = 2
    nrows = int(np.ceil(C_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(C_plot):
        ax = axes_flat[i]
        ax.plot(days, rmse_t_c[:, i], linewidth=1.8)
        ax.set_ylim(0, 5)
        ax.set_title(plot_tokens[i], fontsize=11)
        ax.set_ylabel("RMSE (phys units)")
        ax.set_xlabel("Lead time (days)")
        ax.grid(True, alpha=0.25)

    for j in range(C_plot, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Autoregressive RMSE vs Time (per variable)\n"
        f"split={args.split}  start_index={args.start_index}",
        fontsize=13
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    png_path = os.path.join(args.outdir, f"{base}_rmse_grid.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved RMSE grid plot -> {png_path}")

    # ---------- optional per-variable plots ----------
    if args.save_individual:
        for j, name in enumerate(plot_tokens):
            plt.figure(figsize=(7, 4))
            plt.plot(days, rmse_t_c[:, j], linewidth=2.0)
            plt.ylim(0, 5)
            plt.xlabel("Lead time (days)")
            plt.ylabel("RMSE (phys units)")
            plt.title(f"Autoregressive RMSE vs Time — {name}")
            plt.grid(True, alpha=0.25)
            out_i = os.path.join(args.outdir, f"{base}_rmse_{name}.png")
            plt.tight_layout()
            plt.savefig(out_i, dpi=150)
            plt.close()
        print("[INFO] Saved individual per-variable RMSE plots.")

    # ---------- debug prints ----------
    print("[DEBUG] first-step per-var RMSE:", np.round(rmse_t_c[0], 6))
    print("[DEBUG] last-step  per-var RMSE:", np.round(rmse_t_c[-1], 6))
    print("[DEBUG] overall RMSE first 5:", np.round(rmse_overall[:5], 6))
    print("[DEBUG] overall RMSE last  5:", np.round(rmse_overall[-5:], 6))


if __name__ == "__main__":
    main()
