import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

# ---------------- variable token parsing ----------------
# ERA-style tokens -> MERRA-2 short codes
VAR_MAP: Dict[str, str] = {
    # 3D (has 'lev'):
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
    # 2D (no 'lev'):
    "surface_pressure": "PS",
    "mean_sea_level_pressure": "SLP",
    "surface_geopotential": "PHIS",
}

def _expand_semicolon_groups(tokens: List[str]) -> List[str]:
    """Allow YAML lines like '- a; b; c'. Returns a flat list."""
    out: List[str] = []
    for t in tokens:
        if isinstance(t, str) and ";" in t:
            out.extend([s.strip() for s in t.split(";") if s.strip()])
        else:
            out.append(str(t))
    return out

def parse_token(token: str) -> Tuple[str, Optional[float]]:
    """
    Accepts:
      - 2D names: 'surface_pressure', 'mean_sea_level_pressure', 'surface_geopotential'
      - 3D names with level: 'temperature_500', 'u_component_of_wind_850', ...
    Returns (MERRA_code, level_or_None).
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
        "Use 2D: surface_pressure|mean_sea_level_pressure|surface_geopotential "
        "or 3D like 'temperature_500', 'u_component_of_wind_300', ..."
    )

def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
    """Normalize lon to [-180, 180) and sort by lon if needed."""
    if "lon" in ds.coords:
        lo = ds["lon"]
        if (lo > 180).any():
            ds = ds.assign_coords(lon=((lo + 180) % 360) - 180).sortby("lon")
    return ds

def _ensure_lat_ascending(ds: xr.Dataset) -> xr.Dataset:
    """Sort latitude ascending (many reanalyses store descending)."""
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    if lat_name is None:
        raise KeyError("lat/latitude coordinate not found.")
    if np.any(np.diff(ds[lat_name].values) < 0):
        ds = ds.sortby(lat_name, ascending=True)
    return ds

# ---------------- dataset ----------------
class MERRA2SR(Dataset):
    """
    MERRA-2 (e.g., M2I3NPASM) loader for next-timestep prediction.

    Returns standardized tensors:
      xb: [C,H,W] at time t
      tb: [C,H,W] at time t+1

    Normalization files expected at <root> (keys are *your tokens*):
      normalize_mean.npz, normalize_std.npz

    Directory layout:
      <root>/{train,val,test}/... .nc4 files (symlinks ok)
    """
    def __init__(
        self,
        root: str,
        variables: List[str],
        split: Optional[str] = None,
        engine: str = "netcdf4",
        lat_first: bool = False,
    ):
        super().__init__()
        self.root = root
        self.engine = engine
        self.lat_first = lat_first

        # ---- collect files (follow symlinks) ----
        search_root = os.path.join(root, split) if (split and os.path.isdir(os.path.join(root, split))) else root
        self.files: List[str] = self._collect_nc4_files(search_root)
        if not self.files:
            raise FileNotFoundError(f"No .nc4 files under {search_root}")

        # ---- variable parsing & order ----
        self.variables_tokens = _expand_semicolon_groups(list(variables))
        self.variables_parsed = [parse_token(v) for v in self.variables_tokens]

        # ---- probe first file for grid & level snapping ----
        with xr.open_dataset(self.files[0], engine=self.engine, chunks=None) as probe:
            probe = _ensure_lon_range(_ensure_lat_ascending(probe))

            self.lat_name = "lat" if "lat" in probe.coords else "latitude"
            self.lon_name = "lon" if "lon" in probe.coords else "longitude"

            self._levs = np.array(probe["lev"].values, dtype=float) if "lev" in probe.coords else np.array([])
            self._lev_index: List[Optional[int]] = []
            for (code, lvl) in self.variables_parsed:
                if code not in probe.variables:
                    raise KeyError(f"'{code}' not found in {os.path.basename(self.files[0])}")
                if lvl is None:
                    self._lev_index.append(None)  # 2D var
                else:
                    j = int(np.argmin(np.abs(self._levs - float(lvl))))
                    self._lev_index.append(j)

            H = int(probe.sizes[self.lat_name])
            W = int(probe.sizes[self.lon_name])

        self.C = len(self.variables_tokens)
        self.H, self.W = H, W

        # ---- normalization (by token name) ----
        mean_path = os.path.join(root, "normalize_mean.npz")
        std_path  = os.path.join(root, "normalize_std.npz")
        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            # allow fallback names 'mean.npz'/'std.npz'
            alt_mean, alt_std = os.path.join(root, "mean.npz"), os.path.join(root, "std.npz")
            if os.path.exists(alt_mean) and os.path.exists(alt_std):
                mean_path, std_path = alt_mean, alt_std
            else:
                raise FileNotFoundError(
                    f"Normalization files not found:\n  {mean_path}\n  {std_path}\n"
                    "Compute/train stats first and save with token keys."
                )
        m, s = np.load(mean_path), np.load(std_path)

        def _fetch_stat(npz, name):
            vals = []
            missing = []
            for tok in self.variables_tokens:
                if tok in npz:
                    vals.append(np.float32(npz[tok]))
                else:
                    missing.append(tok)
            if missing:
                raise KeyError(
                    f"{name} missing keys for tokens: {missing}\n"
                    "Recompute stats with the exact same variable list and order."
                )
            arr = np.array(vals, dtype=np.float32).reshape(-1, 1, 1)
            return arr

        self.means = _fetch_stat(m, "normalize_mean.npz")
        self.stds  = np.maximum(_fetch_stat(s, "normalize_std.npz"), 1e-6)

        # ---- build a global (file, t) index so that t+1 always exists ----
        self._index: List[Tuple[int, int]] = []  # (file_idx, t_in_file)
        self._time_cache: List[np.ndarray] = []  # per-file time arrays
        for fi, fp in enumerate(self.files):
            with xr.open_dataset(fp, engine=self.engine, chunks=None) as ds:
                T = int(ds.sizes["time"])
                times = np.array(ds["time"].values)
            self._time_cache.append(times)
        # create pairs across boundaries: for all global t where next exists
        for fi in range(len(self.files)):
            T = len(self._time_cache[fi])
            for ti in range(T):
                # next time:
                nfi, nti = fi, ti + 1
                if nti >= T:
                    nfi += 1
                    nti = 0
                if nfi < len(self.files):
                    self._index.append((fi, ti))
        # now __len__ is len(self._index)

    # ---------------- dataset protocol ----------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        fi, ti = self._index[idx]
        nfi, nti = self._next_pair(fi, ti)

        x_arr = self._read_one_at(self.files[fi],  ti)   # [C,H,W]
        t_arr = self._read_one_at(self.files[nfi], nti)  # [C,H,W]

        xb = self._standardize(x_arr)
        tb = self._standardize(t_arr)
        return torch.from_numpy(xb), torch.from_numpy(tb)

    # ---------------- helpers ----------------
    @staticmethod
    def _collect_nc4_files(root_dir: str) -> List[str]:
        out: List[str] = []
        for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
            for fn in filenames:
                if fn.endswith(".nc4"):
                    out.append(os.path.join(dirpath, fn))
        return sorted(out)

    def _next_pair(self, fi: int, ti: int) -> Tuple[int, int]:
        T = len(self._time_cache[fi])
        nfi, nti = fi, ti + 1
        if nti >= T:
            nfi += 1
            nti = 0
        return nfi, nti

    def _read_one_at(self, fp: str, ti: int) -> np.ndarray:
        """
        Read exact time index ti from file fp and stack selected channels in
        the order of self.variables_tokens. Returns [C,H,W] float32.
        """
        with xr.open_dataset(fp, engine=self.engine, chunks=None) as ds:
            ds = _ensure_lon_range(_ensure_lat_ascending(ds))
            T = int(ds.sizes["time"])
            if not (0 <= ti < T):
                raise IndexError(f"time {ti} out of range [0,{T}) for {os.path.basename(fp)}")

            chans = []
            for (code, _lvl), li in zip(self.variables_parsed, self._lev_index):
                if li is None:
                    da = ds[code].isel(time=ti)              # 2D [lat,lon]
                else:
                    da = ds[code].isel(time=ti, lev=li)      # 3D slice [lat,lon]
                # enforce (lat,lon) order then ndarray
                da = da.transpose(self.lat_name, self.lon_name)
                arr = np.asarray(da.values, dtype=np.float32)
                arr[np.isinf(arr)] = np.nan
                chans.append(arr)
        out = np.stack(chans, axis=0)                        # [C,H,W]
        # NaN hygiene only; standardization happens later
        out = np.where(np.isfinite(out), out, np.nan)
        return out

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        # x: [C,H,W], per-channel z-score with finite-safe handling
        z = (x - self.means) / self.stds
        z = np.clip(z, -10.0, 10.0)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---------------- convenience ----------------
    @property
    def img_resolution(self) -> Tuple[int, int]:
        return (self.H, self.W)

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        with xr.open_dataset(self.files[0], engine=self.engine, chunks=None) as ds:
            ds = _ensure_lon_range(_ensure_lat_ascending(ds))
            lat = np.array(ds[self.lat_name].values, dtype=np.float32)
            lon = np.array(ds[self.lon_name].values, dtype=np.float32)
        return lat, lon

    def get_time(self, idx: int) -> Tuple[np.datetime64, np.datetime64]:
        fi, ti = self._index[idx]
        nfi, nti = self._next_pair(fi, ti)
        t0 = np.datetime64(self._time_cache[fi][ti])
        t1 = np.datetime64(self._time_cache[nfi][nti])
        return t0, t1
