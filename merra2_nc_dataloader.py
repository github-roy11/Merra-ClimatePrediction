import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

# ---------- variable token parsing ----------
# ERA5-style tokens -> MERRA-2 short codes
# 3D on pressure levels (has 'lev'): codes below
# 2D (no 'lev'): PS, SLP, PHIS
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

def _expand_semicolon_groups(tokens):
    """Allow YAML like '- a; b; c'. Returns a flat list of clean tokens."""
    out = []
    for t in tokens:
        if isinstance(t, str) and ";" in t:
            out.extend([s.strip() for s in t.split(";") if s.strip()])
        else:
            out.append(t)
    return out

def parse_token(token: str) -> Tuple[str, Optional[float]]:
    """
    Accepts:
      - full 2D names (no level): e.g. 'surface_pressure', 'mean_sea_level_pressure', 'surface_geopotential'
      - 3D names with a numeric level: e.g. 'temperature_500'
    """
    # 1) Exact 2D token (the whole token is the variable name)
    if token in VAR_MAP:
        return VAR_MAP[token], None

    # 2) 3D token with a trailing numeric level
    if "_" in token:
        base, lvl = token.rsplit("_", 1)
        if base in VAR_MAP:
            try:
                return VAR_MAP[base], float(lvl)
            except ValueError:
                # trailing part isn't numeric -> treat as invalid 3D token
                pass  # fall through to error below

    raise KeyError(
        f"Variable token '{token}' not supported. "
        f"Supported 2D names: {[k for k in VAR_MAP.keys() if k in ['surface_pressure','mean_sea_level_pressure','surface_geopotential']]} "
        f"and 3D patterns like 'temperature_500', 'u_component_of_wind_850', etc."
    )

def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
    """Normalize lon to [-180, 180) and sort if needed."""
    if "lon" in ds.coords:
        lo = ds["lon"]
        if (lo > 180).any():
            ds = ds.assign_coords(lon=((lo + 180) % 360) - 180).sortby("lon")
    return ds


# -------------- Dataset -----------------
class MERRA2SR(Dataset):
    """
    MERRA-2 M2I3NPASM (3-hourly instantaneous) dataloader for time-step prediction.

    Returns (xb, tb) where:
      xb: all selected variables at time t   -> FloatTensor [C, H, W]
      tb: all selected variables at time t+1 -> FloatTensor [C, H, W]

    Normalization files at <root>:
      normalize_mean.npz  and  normalize_std.npz
      Keys must match your tokens (e.g. 'temperature_500', 'surface_pressure', ...).

    Directory layout:
      - <root>/train, <root>/val, <root>/test (can be symlinked years).
      - Set split='train'|'val'|'test' to read under root/split.
      - This loader follows symlinks when collecting files.
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

        # -------- collect files (follow symlinks) --------
        search_root = os.path.join(root, split) if (split and os.path.isdir(os.path.join(root, split))) else root
        self.files: List[str] = self._collect_nc4_files(search_root)
        if not self.files:
            raise FileNotFoundError(f"No .nc4 found under {search_root}. Check paths/symlinks/permissions.")

        # -------- variables + (code,level) parsing --------
        # self.variables_tokens = _expand_semicolon_groups(list(variables))           # keep exact order
        # self.variables_parsed = [parse_token(v) for v in variables]  # -> [(code, level_or_None), ...]
        
        self.variables_tokens = _expand_semicolon_groups(list(variables))  # keep exact order
        self.variables_parsed = [parse_token(v) for v in self.variables_tokens]  # <- fixed

        # -------- probe grid + snap levels --------
        probe = xr.open_dataset(self.files[0], engine=self.engine, chunks=None)
        probe = _ensure_lon_range(probe)

        # available levels
        self._levs = np.array(probe["lev"].values, dtype=float) if "lev" in probe.coords else np.array([])
        # build level index per var; None for 2D vars
        self._lev_index: List[Optional[int]] = []
        for (code, lvl) in self.variables_parsed:
            if code not in probe.variables:
                raise KeyError(f"MERRA-2 file missing variable '{code}'. Check product/content.")
            if lvl is None:
                self._lev_index.append(None)  # 2D var
            else:
                j = int(np.argmin(np.abs(self._levs - float(lvl))))
                self._lev_index.append(j)

        H = int(probe.sizes["lat"])
        W = int(probe.sizes["lon"])
        self.CHW = (len(self.variables_tokens), H, W)
        probe.close()

        # -------- normalization (train stats) --------
        mean_path = os.path.join(root, "normalize_mean.npz")
        std_path  = os.path.join(root, "normalize_std.npz")
        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            raise FileNotFoundError(
                f"Normalization files not found:\n  {mean_path}\n  {std_path}\n"
                f"Compute them first (on train) and place at the split root."
            )
        means_npz = np.load(mean_path)
        stds_npz  = np.load(std_path)
        try:
            self.means = np.stack([means_npz[v] for v in self.variables_tokens], axis=0).reshape(-1, 1, 1).astype(np.float32)
            self.stds  = np.stack([stds_npz[v]  for v in self.variables_tokens], axis=0).reshape(-1, 1, 1).astype(np.float32)
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"normalize_mean/std missing key '{missing}'. Recompute stats with the exact same variable list.") from None

        # cache #timesteps per file (usually 8)
        self._T_cache: Dict[str, int] = {}

    # ------------- utilities -------------
    @staticmethod
    def _collect_nc4_files(root_dir: str) -> List[str]:
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
            for fn in filenames:
                if fn.endswith(".nc4"):
                    out.append(os.path.join(dirpath, fn))
        return sorted(out)

    def _steps_in_file(self, fp: str) -> int:
        T = getattr(self, "_T_cache", {}).get(fp)
        if T is None:
            ds = xr.open_dataset(fp, engine=self.engine, chunks=None)
            T = int(ds.sizes["time"])
            ds.close()
            self._T_cache[fp] = T
        return T
    
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-6  # protect against tiny std
        z = (x - self.means) / (self.stds + eps)
        # clamp extreme z-scores so one bad pixel can't blow up activations
        z = np.clip(z, -10.0, 10.0)
        # ensure no NaN/Inf survives
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ------------- dataset protocol -------------
    def __len__(self) -> int:
        """
        Number of valid (t, t+1) pairs across all files.
        Uses T of the first file (M2I3NPASM is consistent: T=8).
        Returns total_steps - 1 so that t+1 always exists.
        """
        T0 = self._steps_in_file(self.files[0])
        total_steps = len(self.files) * T0
        return max(total_steps - 1, 0)

    def _read_one_at(self, fp: str, ti: int) -> np.ndarray:
        """
        Read exact time index ti from file fp, stacking selected variables (2D+3D).
        Returns np.ndarray [C, H, W] (float32).
        """
        ds = xr.open_dataset(fp, engine=self.engine, chunks=None)
        ds = _ensure_lon_range(ds)
        T = int(ds.sizes["time"])
        if not (0 <= ti < T):
            ds.close()
            raise IndexError(f"time {ti} out of range [0,{T}) for {os.path.basename(fp)}")

        chans = []
        for (code, _lvl), li in zip(self.variables_parsed, self._lev_index):
            if li is None:
                # 2D variable, just pick time slice
                da = ds[code].isel(time=ti)          # [lat, lon]
            else:
                # 3D variable, pick time and snapped level
                da = ds[code].isel(time=ti, lev=li)  # [lat, lon]
            arr = np.asarray(da.values, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            chans.append(arr)
        ds.close()
        return np.stack(chans, axis=0)  # [C, H, W]

    def __getitem__(self, idx: int):
        """
        Time-step prediction:
          xb = variables at time t   -> [C,H,W]
          tb = variables at time t+1 -> [C,H,W]
        DataLoader will batch these to [B,C,H,W].
        """
        C, H, W = self.CHW
        T = self._steps_in_file(self.files[0])

        file_i = idx // T
        t_in   = idx % T

        # next time (possibly next file)
        next_file_i = file_i
        next_t_in   = t_in + 1
        if next_t_in >= T:
            next_t_in = 0
            next_file_i = file_i + 1

        fp_x = self.files[file_i]
        fp_t = self.files[next_file_i]

        x_arr = self._read_one_at(fp_x, t_in)        # [C,H,W]
        t_arr = self._read_one_at(fp_t, next_t_in)   # [C,H,W]

        # standardize (train stats)
        x_std = self._standardize(x_arr)
        t_std = self._standardize(t_arr)

        xb = torch.from_numpy(self._standardize(x_arr))  # [C,H,W]
        tb = torch.from_numpy(self._standardize(t_arr))  # [C,H,W]
        return xb, tb

    # ------------- convenience -------------
    @property
    def img_resolution(self) -> Tuple[int, int]:
        _, H, W = self.CHW
        return H, W

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        ds = xr.open_dataset(self.files[0], engine=self.engine, chunks=None)
        lat = np.array(ds["lat"].values, dtype=np.float32)
        lon = np.array(ds["lon"].values, dtype=np.float32)
        ds.close()
        return lat, lon

    def get_time(self, idx: int) -> Tuple[np.datetime64, np.datetime64]:
        """Return (time_t, time_t+1) for the given sample index."""
        T = self._steps_in_file(self.files[0])
        file_i = idx // T
        t_in   = idx % T

        next_file_i = file_i
        next_t_in   = t_in + 1
        if next_t_in >= T:
            next_t_in = 0
            next_file_i = file_i + 1

        ds0 = xr.open_dataset(self.files[file_i], engine=self.engine, chunks=None)
        ts0 = np.datetime64(ds0["time"].values[t_in])
        ds0.close()

        ds1 = xr.open_dataset(self.files[next_file_i], engine=self.engine, chunks=None)
        ts1 = np.datetime64(ds1["time"].values[next_t_in])
        ds1.close()
        return ts0, ts1


# ------------- quick self-test -------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ROOT = "/storage/home/sbr5878/scratch/new_diffusion/MERRA2_splits"
    # start small; expand once normalization is computed for your full list
    VARS = [
        "temperature_500",
        "u_component_of_wind_500",
        "v_component_of_wind_500",
        "specific_humidity_500",
        "surface_pressure",              # 2D example
        "mean_sea_level_pressure",       # 2D example
        "surface_geopotential",          # 2D example
    ]

    ds = MERRA2SR(root=ROOT, variables=VARS, split="train")
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

    xb, tb = next(iter(dl))
    print("xb:", xb.shape, "tb:", tb.shape)  # expect both [8, C, H, W]
    print("mean xb:", xb.mean().item(), "std xb:", xb.std().item())
    print("mean tb:", tb.mean().item(), "std tb:", tb.std().item())
