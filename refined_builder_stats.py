#!/usr/bin/env python
"""
Compute per-variable mean/std for MERRA-2 (train split) and save:
  - <root>/normalize_mean.npz
  - <root>/normalize_std.npz

Tokens can be passed exactly like in your YAML, including semicolon-separated groups.
We compute stats on the **raw fields** (no standardization) across all time/lat/lon
and across all files under <root>/train (or a custom split dir).

Usage:
  python refined_builder_stats.py \
    --root /storage/home/sbr5878/scratch/diffusion/MERRA2_splits \
    --split train \
    --variables "
      epv_1000; epv_925; epv_850; epv_700; epv_600; epv_500; epv_400; epv_300; epv_250; epv_200; epv_150; epv_100; epv_70; epv_50;
      geopotential_height_1000; geopotential_height_925; ... ; surface_geopotential; surface_pressure; mean_sea_level_pressure
    "

Notes:
- Stats must be computed using the **72 input tokens** in the exact order you will use for training.
- We ignore NaNs and Infs and compute a global mean/std across all time steps & all pixels.
"""

import os
import argparse
from typing import List, Tuple, Optional, Dict
import numpy as np
import xarray as xr

# ---------- variable token parsing (same as your dataloader) ----------
VAR_MAP: Dict[str, str] = {
    # 3D:
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity": "QV",
    "relative_humidity": "RH",
    "omega": "OMEGA",
    "geopotential_height": "H",
    "cloud_ice": "QI",
    "cloud_liquid": "QL",
    "epv": "EPV",
    "ozone": "O3",
    # 2D:
    "surface_pressure": "PS",
    "mean_sea_level_pressure": "SLP",
    "surface_geopotential": "PHIS",
}

def _expand_semicolon_groups(tokens_or_string) -> List[str]:
    """
    Allow passing variables as a single string with ';' separators
    or as a list whose elements may themselves contain ';'.
    Returns a flat, stripped list.
    """
    if isinstance(tokens_or_string, str):
        parts = [p.strip() for p in tokens_or_string.split(";") if p.strip()]
        return parts
    out = []
    for t in tokens_or_string:
        if isinstance(t, str) and ";" in t:
            out.extend([s.strip() for s in t.split(";") if s.strip()])
        else:
            out.append(t)
    return out

def parse_token(token: str) -> Tuple[str, Optional[float]]:
    """
    'surface_pressure' -> ('PS', None)   # 2D
    'temperature_500'  -> ('T', 500.0)   # 3D @ level nearest to 500
    """
    if token in VAR_MAP:
        return VAR_MAP[token], None
    if "_" in token:
        base, lvl = token.rsplit("_", 1)
        if base in VAR_MAP:
            try:
                return VAR_MAP[base], float(lvl)
            except ValueError:
                pass
    raise KeyError(
        f"Unsupported token '{token}'. "
        "Use 2D names (surface_pressure, mean_sea_level_pressure, surface_geopotential) "
        "or 3D patterns like 'temperature_500', 'u_component_of_wind_850', etc."
    )

def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
    """Normalize lon to [-180, 180) and sort if needed."""
    if "lon" in ds.coords:
        lo = ds["lon"]
        if (lo > 180).any():
            ds = ds.assign_coords(lon=((lo + 180) % 360) - 180).sortby("lon")
    return ds

# ---------- streaming mean/std (Welford) ----------
class OnlineMeanStd:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: np.ndarray):
        """
        x: arbitrary shape, will be flattened; NaN/Inf ignored.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        # mask non-finites
        m = np.isfinite(x)
        if not m.any():
            return
        vals = x[m]
        # batch update
        count = vals.size
        mean_x = float(vals.mean())
        # sum of squared diffs from its mean
        M2_x = float(((vals - mean_x) ** 2).sum())
        # combine
        if self.n == 0:
            self.n = count
            self.mean = mean_x
            self.M2 = M2_x
        else:
            n_a, n_b = self.n, count
            delta = mean_x - self.mean
            self.mean = self.mean + delta * (n_b / (n_a + n_b))
            self.M2 = self.M2 + M2_x + delta * delta * (n_a * n_b / (n_a + n_b))
            self.n = n_a + n_b

    def finalize(self) -> Tuple[float, float]:
        if self.n < 2:
            return float(self.mean), 0.0
        var = self.M2 / (self.n - 1)
        var = max(var, 0.0)
        return float(self.mean), float(np.sqrt(var))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing split dirs and where NPZs will be saved")
    ap.add_argument("--split", default="train", help="Split to scan for stats (default: train)")
    ap.add_argument("--variables", nargs="+", required=True,
                    help="Tokens (72 inputs). You may pass a single quoted string with ';' separators.")
    ap.add_argument("--engine", default="netcdf4")
    args = ap.parse_args()

    tokens = _expand_semicolon_groups(args.variables)
    print(f"[info] #tokens={len(tokens)}")
    for i, t in enumerate(tokens):
        print(f"  [{i:02d}] {t}")

    # Collect files in split dir (follow symlinks)
    split_dir = os.path.join(args.root, args.split)
    search_root = split_dir if os.path.isdir(split_dir) else args.root
    nc_files = []
    for dirpath, _, filenames in os.walk(search_root, followlinks=True):
        for fn in filenames:
            if fn.endswith(".nc4"):
                nc_files.append(os.path.join(dirpath, fn))
    nc_files = sorted(nc_files)
    if not nc_files:
        raise FileNotFoundError(f"No .nc4 files found under {search_root}")

    # Open one file to probe levels & grid
    probe = xr.open_dataset(nc_files[0], engine=args.engine, chunks=None)
    probe = _ensure_lon_range(probe)
    H = int(probe.sizes["lat"])
    W = int(probe.sizes["lon"])
    levs = np.array(probe["lev"].values, dtype=float) if "lev" in probe.coords else np.array([])
    # Build (code, level_idx or None)
    parsed = []
    for t in tokens:
        code, lvl = parse_token(t)
        if code not in probe.variables:
            raise KeyError(f"Variable '{code}' not found in {os.path.basename(nc_files[0])}")
        if lvl is None:
            parsed.append((code, None))
        else:
            j = int(np.argmin(np.abs(levs - float(lvl))))
            parsed.append((code, j))
    probe.close()

    # One OnlineMeanStd per token
    stats = [OnlineMeanStd() for _ in tokens]

    # Stream through files and time steps
    for fi, fp in enumerate(nc_files):
        ds = xr.open_dataset(fp, engine=args.engine, chunks=None)
        ds = _ensure_lon_range(ds)
        T = int(ds.sizes["time"])
        print(f"[scan] {fi+1}/{len(nc_files)} {os.path.basename(fp)} (T={T})")
        for ti in range(T):
            # pull each token slice
            for k, ((code, lev_idx), tok) in enumerate(zip(parsed, tokens)):
                if lev_idx is None:
                    da = ds[code].isel(time=ti)          # [lat, lon]
                else:
                    da = ds[code].isel(time=ti, lev=lev_idx)  # [lat, lon]
                arr = np.asarray(da.values, dtype=np.float64)
                # sanitize
                arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)  # keep NaN so OnlineMeanStd drops them
                stats[k].update(arr)
        ds.close()

    # Finalize and save
    means = {}
    stds  = {}
    for tok, s in zip(tokens, stats):
        m, sd = s.finalize()
        # guard tiny/degenerate std
        sd = float(sd) if sd > 1e-8 else 1e-8
        means[tok] = np.float32(m)
        stds[tok]  = np.float32(sd)

    mean_path = os.path.join(args.root, "normalize_mean.npz")
    std_path  = os.path.join(args.root, "normalize_std.npz")
    np.savez_compressed(mean_path, **means)
    np.savez_compressed(std_path,  **stds)
    print(f"[done] wrote:\n  {mean_path}\n  {std_path}")

if __name__ == "__main__":
    main()