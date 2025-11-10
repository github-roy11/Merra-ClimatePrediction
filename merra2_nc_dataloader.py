# import os
# from glob import glob
# from typing import Tuple, List

# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import xarray as xr

# #  -------- variable token parsing --------
# VAR_MAP = {
#     "temperature": "T",
#     "u_component_of_wind": "U",
#     "v_component_of_wind": "V",
#     "specific_humidity": "QV",
#     # NOTE: geopotential & many surface fields are NOT in M2I3NPASM.
# }

# def parse_token(token: str) -> Tuple[str, float]:
#     # "temperature_500" -> ("T", 500.0)
#     if "_" not in token:
#         raise ValueError(f"Expected <name>_<level>, got '{token}'")
#     name, lvl = token.rsplit("_", 1)
#     if name not in VAR_MAP:
#         raise KeyError(f"Variable '{name}' not supported in M2I3NPASM. "
#                        f"Supported: {list(VAR_MAP.keys())}")
#     return VAR_MAP[name], float(lvl)

# def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
#     if "lon" in ds.coords:
#         lo = ds["lon"]
#         if (lo > 180).any():
#             ds = ds.assign_coords(lon=((lo + 180) % 360) - 180).sortby("lon")
#     return ds

# # -------------- Dataset -----------------

# class MERRA2SR(Dataset):
#     """
#     Drop-in replacement for ERA5SR but reading MERRA-2 M2I3NPASM .nc4 files.

#     variables tokens like: ["temperature_500", "u_component_of_wind_850", ...]
#     Returns (x, t) where:
#       - x: last channel (1, H, W)
#       - t: all previous channels (C-1, H, W)

#     Needs normalization files in <root>:
#       normalize_mean.npz  /  normalize_std.npz
#     with keys exactly matching your tokens.
#     """

#     def __init__(
#         self,
#         root: str,
#         variables: List[str],
#         conditions: int = 1,
#         split: str = None,
#         rescale: int = 1,
#         engine: str = "netcdf4",
#         lat_first: bool = False
#     ):
#         super().__init__()
#         self.root = root
#         self.engine = engine
#         self.lat_first = lat_first

#         # Files
#         # search_root = os.path.join(root, split) if (split and os.path.isdir(os.path.join(root, split))) else root
#         # self.files = sorted(glob(os.path.join(search_root, "**", "*.nc4"), recursive=True))
#         # if not self.files:
#         #     raise FileNotFoundError(f"No .nc4 under {search_root}")
#         search_root = os.path.join(root, split) if (split and os.path.isdir(os.path.join(root, split))) else root

#         def _collect_nc4_files(root_dir: str):
#             out = []
#             for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
#                 for fn in filenames:
#                     if fn.endswith(".nc4"):
#                         out.append(os.path.join(dirpath, fn))
#             return sorted(out)

#         self.files = _collect_nc4_files(search_root)   # <<< add this line
#         if not self.files:
#             raise FileNotFoundError(f"No .nc4 under {search_root} (symlinks? perms?)")
        
#         # Variables
#         self.variables_tokens = list(variables)
#         self.variables_parsed = [parse_token(v) for v in variables]  # -> [('T',500), ('U',850), ...]

#         # Normalization
#         mean_path = os.path.join(root, "normalize_mean.npz")
#         std_path  = os.path.join(root, "normalize_std.npz")
#         if not (os.path.exists(mean_path) and os.path.exists(std_path)):
#             raise FileNotFoundError(
#                 f"Normalization files not found:\n  {mean_path}\n  {std_path}\n"
#                 f"Run your normalization helper first."
#             )
#         means_npz = np.load(mean_path)
#         stds_npz  = np.load(std_path)
#         try:
#             self.means = np.stack([means_npz[v] for v in self.variables_tokens], axis=0).reshape(-1,1,1)
#             self.stds  = np.stack([stds_npz[v]  for v in self.variables_tokens], axis=0).reshape(-1,1,1)
#         except KeyError as e:
#             missing = str(e).strip("'")
#             raise KeyError(f"normalize_mean/std missing key '{missing}'. "
#                            f"Recompute stats with the exact same variable list.") from None

#         # Probe one file: shape + nearest pressure levels to use
#         probe = xr.open_dataset(self.files[0], engine=self.engine, chunks=None)
#         probe = _ensure_lon_range(probe)
#         for (code, _) in self.variables_parsed:
#             if code not in probe.variables:
#                 raise KeyError(f"MERRA-2 file missing variable '{code}'. "
#                                f"Check product (M2I3NPASM has T,U, V, QV).")
#         levs_avail = np.array(probe["lev"].values).astype(float)
#         self._lev_index = []
#         for (_, lvl) in self.variables_parsed:
#             idx = int(np.argmin(np.abs(levs_avail - float(lvl))))
#             self._lev_index.append(idx)
#         # H = probe.dims["lat"]; W = probe.dims["lon"]
#         H = probe.sizes["lat"]; W = probe.sizes["lon"]
#         self.shape = (len(self.variables_tokens), H, W)
#         probe.close()

#         self.conditions = conditions
#         self.scale = rescale

#     @property
#     def n_channels(self):
#         return self.shape[0] * self.scale - self.conditions

#     @property
#     def cond_channels(self):
#         return self.conditions

#     @property
#     def img_resolution(self):
#         return self.shape[1], self.shape[2]

#     def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
#         ds = xr.open_dataset(self.files[0], engine=self.engine, chunks=None)
#         lat = np.array(ds["lat"].values, dtype=np.float32)
#         lon = np.array(ds["lon"].values, dtype=np.float32)
#         ds.close()
#         return lat, lon

#     def get_time(self, idx: int) -> np.datetime64:
#         file_idx = idx % len(self.files)
#         ds = xr.open_dataset(self.files[file_idx], engine=self.engine, chunks=None)
#         tvals = ds["time"].values
#         ti = (idx // len(self.files)) % len(tvals)
#         ts = np.datetime64(tvals[ti])
#         ds.close()
#         return ts

#     def __len__(self) -> int:
#         # MERRA-2 inst3_3d files typically have 8 time steps/day; this is a simple estimate
#         return len(self.files) * 8

#     def _read_one(self, fp: str, time_index: int) -> np.ndarray:
#         ds = xr.open_dataset(fp, engine=self.engine, chunks=None)
#         ds = _ensure_lon_range(ds)
#         # ti = int(time_index % ds.dims["time"])
#         ti = int(time_index % ds.sizes["time"])

#         chans = []
#         for (code, _), lev_idx in zip(self.variables_parsed, self._lev_index):
#             da = ds[code].isel(time=ti, lev=lev_idx)  # 2D [lat, lon]
#             arr = np.array(da.values, dtype=np.float32)
#             chans.append(arr)
#         ds.close()
#         x = np.stack(chans, axis=0)  # [C, H, W]
#         return x

#     def _standardize(self, x: np.ndarray) -> np.ndarray:
#         return (x - self.means) / self.stds

#     def __getitem__(self, idx: int):
#         file_idx = idx % len(self.files)
#         time_in_file = (idx // len(self.files))
#         fp = self.files[file_idx]

#         x_all = self._standardize(self._read_one(fp, time_in_file))  # [C,H,W]
#         t = torch.from_numpy(x_all)
#         x = t[-1:]          # last channel is target
#         return x, t[:-1]    # (1,H,W), (C-1,H,W)


########################################################## Running code ##########################################################

# import os
# from typing import Tuple, List, Dict

# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import xarray as xr

# # ---------- variable token parsing ----------
# # Tokens like "temperature_500" map to MERRA-2 codes + pressure level
# VAR_MAP: Dict[str, str] = {
#     "temperature": "T",
#     "u_component_of_wind": "U",
#     "v_component_of_wind": "V",
#     "specific_humidity": "QV",
#     # NOTE: surface/geopotential are not in M2I3NPASM.
# }

# def parse_token(token: str) -> Tuple[str, float]:
#     """
#     'temperature_500' -> ('T', 500.0)
#     """
#     if "_" not in token:
#         raise ValueError(f"Expected <name>_<level>, got '{token}'")
#     name, lvl = token.rsplit("_", 1)
#     if name not in VAR_MAP:
#         raise KeyError(
#             f"Variable '{name}' not supported in M2I3NPASM. "
#             f"Supported: {list(VAR_MAP.keys())}"
#         )
#     return VAR_MAP[name], float(lvl)

# def _ensure_lon_range(ds: xr.Dataset) -> xr.Dataset:
#     """Make lon in [-180, 180) and sort, if needed."""
#     if "lon" in ds.coords:
#         lo = ds["lon"]
#         if (lo > 180).any():
#             ds = ds.assign_coords(lon=((lo + 180) % 360) - 180).sortby("lon")
#     return ds


# # -------------- Dataset -----------------
# class MERRA2SR(Dataset):
#     """
#     MERRA-2 M2I3NPASM (3-hourly instantaneous) dataloader for time-step prediction.

#     Returns (xb, tb) where:
#       xb: all selected variables at time t   -> FloatTensor [C, H, W]
#       tb: all selected variables at time t+1 -> FloatTensor [C, H, W]

#     Normalization:
#       <root>/normalize_mean.npz  and  <root>/normalize_std.npz
#       with keys that match the variable tokens you pass (e.g. 'temperature_500').

#     Directory layout:
#       - root may be a split directory with symlinks (train/val/test) or the RAW tree.
#       - Set split='train'/'val'/'test' to read under root/split.
#       - This loader follows symlinks when collecting files.
#     """

#     def __init__(
#         self,
#         root: str,
#         variables: List[str],
#         split: str | None = None,
#         engine: str = "netcdf4",
#         lat_first: bool = False,
#     ):
#         super().__init__()
#         self.root = root
#         self.engine = engine
#         self.lat_first = lat_first

#         # -------- collect files (follow symlinks) --------
#         search_root = os.path.join(root, split) if (split and os.path.isdir(os.path.join(root, split))) else root

#         def _collect_nc4_files(root_dir: str) -> List[str]:
#             out: List[str] = []
#             for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
#                 for fn in filenames:
#                     if fn.endswith(".nc4"):
#                         out.append(os.path.join(dirpath, fn))
#             return sorted(out)

#         self.files: List[str] = _collect_nc4_files(search_root)
#         if not self.files:
#             raise FileNotFoundError(f"No .nc4 found under {search_root}. Check paths/symlinks/permissions.")

#         # -------- variables + nearest pressure levels --------
#         self.variables_tokens = list(variables)
#         self.variables_parsed = [parse_token(v) for v in variables]  # [('T',500), ('U',500), ...]

#         # -------- normalization (train stats) --------
#         mean_path = os.path.join(root, "normalize_mean.npz")
#         std_path  = os.path.join(root, "normalize_std.npz")
#         if not (os.path.exists(mean_path) and os.path.exists(std_path)):
#             raise FileNotFoundError(
#                 f"Normalization files not found:\n  {mean_path}\n  {std_path}\n"
#                 f"Compute them first (on train) and place at the split root."
#             )
#         means_npz = np.load(mean_path)
#         stds_npz  = np.load(std_path)
#         try:
#             self.means = np.stack([means_npz[v] for v in self.variables_tokens], axis=0).reshape(-1, 1, 1).astype(np.float32)
#             self.stds  = np.stack([stds_npz[v]  for v in self.variables_tokens], axis=0).reshape(-1, 1, 1).astype(np.float32)
#         except KeyError as e:
#             missing = str(e).strip("'")
#             raise KeyError(f"normalize_mean/std missing key '{missing}'. Recompute stats with the exact same variable list.") from None

#         # -------- probe grid + level indices --------
#         probe = xr.open_dataset(self.files[0], engine=self.engine, chunks=None)
#         probe = _ensure_lon_range(probe)
#         for (code, _) in self.variables_parsed:
#             if code not in probe.variables:
#                 raise KeyError(f"MERRA-2 file missing variable '{code}'. Check product (M2I3NPASM has T,U,V,QV).")
#         levs_avail = np.array(probe["lev"].values, dtype=float)
#         self._lev_index: List[int] = []
#         for (_, lvl) in self.variables_parsed:
#             self._lev_index.append(int(np.argmin(np.abs(levs_avail - float(lvl)))))

#         H = int(probe.sizes["lat"])
#         W = int(probe.sizes["lon"])
#         self.CHW = (len(self.variables_tokens), H, W)
#         probe.close()

#         # cache #timesteps per file (usually 8)
#         self._T_cache: Dict[str, int] = {}

#     # ------------- handy properties -------------
#     @property
#     def img_resolution(self) -> Tuple[int, int]:
#         _, H, W = self.CHW
#         return H, W

#     def _steps_in_file(self, fp: str) -> int:
#         """Return # of time steps in file (cached)."""
#         T = self._T_cache.get(fp)
#         if T is None:
#             ds = xr.open_dataset(fp, engine=self.engine, chunks=None)
#             T = int(ds.sizes["time"])
#             ds.close()
#             self._T_cache[fp] = T
#         return T

#     def __len__(self) -> int:
#         """
#         Number of valid (t, t+1) pairs across all files.
#         We assume constant T across files (typical for M2I3NPASM: T=8).
#         This returns total_steps - 1 so that t+1 always exists.
#         """
#         T0 = self._steps_in_file(self.files[0])
#         total_steps = len(self.files) * T0
#         return max(total_steps - 1, 0)

#     # ------------- low-level readers -------------
#     def _read_one_at(self, fp: str, ti: int) -> np.ndarray:
#         """
#         Read EXACT time index ti from fp, stack selected variables at chosen levels.
#         Returns np.ndarray [C, H, W] (float32).
#         """
#         ds = xr.open_dataset(fp, engine=self.engine, chunks=None)
#         ds = _ensure_lon_range(ds)
#         T = int(ds.sizes["time"])
#         if not (0 <= ti < T):
#             ds.close()
#             raise IndexError(f"time {ti} out of range [0,{T}) for {os.path.basename(fp)}")

#         chans = []
#         for (code, _), lev_idx in zip(self.variables_parsed, self._lev_index):
#             da = ds[code].isel(time=ti, lev=lev_idx)  # 2D [lat, lon]
#             arr = np.asarray(da.values, dtype=np.float32)
#             chans.append(arr)
#         ds.close()
#         return np.stack(chans, axis=0)  # [C, H, W]

#     # ------------- public helpers -------------
#     def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
#         ds = xr.open_dataset(self.files[0], engine=self.engine, chunks=None)
#         lat = np.array(ds["lat"].values, dtype=np.float32)
#         lon = np.array(ds["lon"].values, dtype=np.float32)
#         ds.close()
#         return lat, lon

#     def get_time(self, idx: int) -> Tuple[np.datetime64, np.datetime64]:
#         """
#         Returns (time_t, time_t+1) for the given sample index.
#         """
#         T = self._steps_in_file(self.files[0])
#         file_i = idx // T
#         t_in   = idx % T
#         # ensure t+1 exists (since __len__ guarantees total-1)
#         next_file_i = file_i
#         next_t_in   = t_in + 1
#         if next_t_in >= T:
#             next_t_in = 0
#             next_file_i = file_i + 1

#         ds0 = xr.open_dataset(self.files[file_i], engine=self.engine, chunks=None)
#         ts0 = np.datetime64(ds0["time"].values[t_in])
#         ds0.close()

#         ds1 = xr.open_dataset(self.files[next_file_i], engine=self.engine, chunks=None)
#         ts1 = np.datetime64(ds1["time"].values[next_t_in])
#         ds1.close()
#         return ts0, ts1

#     # ------------- main fetch -------------
#     def __getitem__(self, idx: int):
#         """
#         Time-step prediction:
#           xb = all variables at time t
#           tb = all variables at time t+1
#         Shapes (per-sample): both [C, H, W] -> DataLoader yields [B, C, H, W].
#         """
#         C, H, W = self.CHW
#         T = self._steps_in_file(self.files[0])

#         file_i = idx // T
#         t_in   = idx % T

#         # next time (may be next file)
#         next_file_i = file_i
#         next_t_in   = t_in + 1
#         if next_t_in >= T:
#             next_t_in = 0
#             next_file_i = file_i + 1

#         fp_x = self.files[file_i]
#         fp_t = self.files[next_file_i]

#         x_arr = self._read_one_at(fp_x, t_in)        # [C,H,W]
#         t_arr = self._read_one_at(fp_t, next_t_in)   # [C,H,W]

#         # standardize with train stats
#         x_std = (x_arr - self.means) / self.stds
#         t_std = (t_arr - self.means) / self.stds

#         xb = torch.from_numpy(x_std)  # [C,H,W]
#         tb = torch.from_numpy(t_std)  # [C,H,W]
#         return xb, tb
    
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader

#     ROOT = "/storage/home/sbr5878/scratch/diffusion/MERRA2_splits"
#     VARS = ["temperature_500","u_component_of_wind_500","v_component_of_wind_500","specific_humidity_500"]

#     ds = MERRA2SR(root=ROOT, variables=VARS, split="train")
#     dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

#     xb, tb = next(iter(dl))
#     print("xb:", xb.shape, "tb:", tb.shape)  # expect both [8, 4, H, W]
#     print("mean xb:", xb.mean().item(), "std xb:", xb.std().item())
#     print("mean tb:", tb.mean().item(), "std tb:", tb.std().item())






##################################### New code with new variable mapping #####################################
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