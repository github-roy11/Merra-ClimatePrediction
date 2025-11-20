#!/usr/bin/env python
import os, math, time, json, argparse, yaml
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- your dataloader ---
from merra2_nc_dataloader import MERRA2SR

# --- your model ---
# Expect a class ForecastModel(in_channels, out_channels, img_resolution, **kwargs)
from model import ForecastModel


# --------- utils ---------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def nanmean(x: torch.Tensor, dim=None, keepdim=False):
    mask = torch.isfinite(x)
    x = torch.where(mask, x, torch.zeros_like(x))
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    return x.sum(dim=dim, keepdim=keepdim) / denom

def center_crop_hw(x: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    # x: [B, C, H, W]
    H, W = x.shape[-2:]
    if H == H_out and W == W_out:
        return x
    top  = max((H - H_out) // 2, 0)
    left = max((W - W_out) // 2, 0)
    return x[..., top:top+H_out, left:left+W_out]

########################## Regarding DDP util functions ##########################

def ddp_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_active() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_active() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def ddp_setup():
    """
    Initialize torch.distributed using environment variables (works with torchrun/srun).
    Also sets the CUDA device from LOCAL_RANK.
    """
    if ddp_is_active():
        return int(os.environ.get("LOCAL_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "nccl"), init_method="env://")
    return local_rank

def ddp_cleanup():
    if ddp_is_active():
        dist.barrier()
        dist.destroy_process_group()

@torch.no_grad()
def reduce_mean(x: torch.Tensor) -> float:
    if not ddp_is_active():
        return float(x.item())
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= get_world_size()
    return float(x.item())

# --------- config ---------
@dataclass
class DataCfg:
    root_dir: str
    inputs: List[str]
    targets: List[str]  # can be same as inputs for full next-step prediction
    batch_size: int = 8
    num_workers: int = 1
    pin_memory: bool = True

@dataclass
class ModelCfg:
    img_resolution: Tuple[int, int] = (361, 576)  # MERRA-2 M2I3NPASM grid
    # extra kwargs passed to ForecastModel; adapt to your model.py
    dim: int = 256
    depth: int = 8
    heads: int = 8
    patch_size: Tuple[int, int] = (1, 2)
    window_size: Tuple[int, int] = (4, 8)
    flash: bool = True

@dataclass
class TrainCfg:
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    ema_decay: float = 0.999
    ema_warmup_epochs: int = 5
    amp: bool = True
    log_every: int = 50
    val_every_epochs: int = 1
    ckpt_dir: str = "./checkpoints"
    run_name: str = "merra2_nextstep"
    resume: Optional[str] = None
    accum_steps: int = 1   # <--- add this

@dataclass
class Cfg:
    data: DataCfg
    model: ModelCfg
    train: TrainCfg


# --------- EMA ---------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k, p in model.state_dict().items() if p.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)


# --------- dataset builders ---------
# def make_loader(root: str, variables: List[str], split: str, batch: int, workers: int, pin: bool):
#     ds = MERRA2SR(root=root, variables=variables, split=split)
#     kwargs = dict(
#         batch_size=batch,
#         shuffle=(split == "train"),
#         num_workers=workers,
#         pin_memory=pin,
#         persistent_workers=(workers > 0),
#     )
#     if workers > 0:
#         kwargs["prefetch_factor"] = 2  # per instructor
#     return ds, DataLoader(ds, **kwargs)

def make_loader(root: str, variables: List[str], split: str, batch: int, workers: int, pin: bool):
    ds = MERRA2SR(root=root, variables=variables, split=split)
    sampler = None
    if ddp_is_active():
        sampler = DistributedSampler(ds, shuffle=(split == "train"), drop_last=False)
    kwargs = dict(
        batch_size=batch,
        shuffle=(sampler is None and split == "train"),
        num_workers=workers,
        pin_memory=pin,
        sampler=sampler,
        persistent_workers=(workers > 0),
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return ds, DataLoader(ds, **kwargs)

# --------- loss (masked mean MSE) ---------
def mse_nan_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error over finite elements of BOTH pred and target.
    This prevents NaN loss if the model temporarily outputs NaN/Inf.
    """
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        # fall back: return 0 to skip this step safely
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff2 = (pred - target) ** 2
    diff2 = torch.where(mask, diff2, torch.zeros_like(diff2))
    return diff2.sum() / mask.sum()

# --------- train/val loops ---------
def run_val(model, dl, device, img_hw: Tuple[int, int], scaler=None):
    model.eval()
    Hc, Wc = img_hw
    losses = []
    with torch.no_grad():
        for xb, tb in dl:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)

            xb = center_crop_hw(xb, Hc, Wc)
            tb = center_crop_hw(tb, Hc, Wc)

            y = model(xb)
            loss = mse_nan_mean(y, tb)
            losses.append(loss.item())

    # DDP-safe average (all ranks participate)
    avg = torch.tensor(sum(losses) / max(len(losses), 1), device=device, dtype=torch.float32)
    if ddp_is_active():
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        avg /= get_world_size()
    return float(avg.item())

def _expand_semicolon_groups(groups):
    """Turn ['a; b; c', 'd; e'] into ['a','b','c','d','e']."""
    flat = []
    for g in groups or []:
        if isinstance(g, str):
            flat.extend([t.strip() for t in g.split(";") if t.strip()])
        else:
            # if someone passed a list accidentally, still split/flatten
            for t in g:
                flat.extend([s.strip() for s in str(t).split(";") if s.strip()])
    return flat

def save_ckpt(path, model, opt, scaler, epoch, step, ema: Optional[EMA], cfg: Cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "epoch": epoch,
        "step": step,
        "cfg": asdict(cfg),
        "ema": (ema.shadow if ema is not None else None),
    }
    torch.save(payload, path)

def load_ckpt(path, model, opt=None, scaler=None) -> Tuple[int,int,Optional[EMA],Optional[Cfg]]:
    chk = torch.load(path, map_location="cpu")
    model.load_state_dict(chk["model"], strict=True)
    if opt is not None and "opt" in chk and chk["opt"] is not None:
        opt.load_state_dict(chk["opt"])
    if scaler is not None and "scaler" in chk and chk["scaler"] is not None:
        scaler.load_state_dict(chk["scaler"])
    ema = None
    if "ema" in chk and chk["ema"] is not None:
        ema = EMA(model, decay=0.0)  # dummy decay; we only need storage
        ema.shadow = chk["ema"]
    cfg = None
    if "cfg" in chk and chk["cfg"] is not None:
        cfg = Cfg(
            data=DataCfg(**chk["cfg"]["data"]),
            model=ModelCfg(**chk["cfg"]["model"]),
            train=TrainCfg(**chk["cfg"]["train"]),
        )
    return chk.get("epoch", 0), chk.get("step", 0), ema, cfg

# --------- main ---------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = p.parse_args()
    
    # --- DDP init ---
    local_rank = 0
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --- defaults (you can override via YAML) ---
    default_cfg = Cfg(
        data=DataCfg(
            root_dir="/storage/home/sbr5878/scratch/new_diffusion/MERRA2_splits",
            inputs=[
                "temperature_500",
                "u_component_of_wind_500",
                "v_component_of_wind_500",
                "specific_humidity_500",
                "surface_pressure",
                "mean_sea_level_pressure",
                "surface_geopotential",
            ],
            targets=[
                # next-time prediction for the SAME channels (adjust if you want to drop PHIS from targets)
                "temperature_500",
                "u_component_of_wind_500",
                "v_component_of_wind_500",
                "specific_humidity_500",
                "surface_pressure",
                "mean_sea_level_pressure",
                "surface_geopotential",
            ],
            batch_size=8,
            num_workers=1,
            pin_memory=True,
        ),
        model=ModelCfg(
            img_resolution=(361, 576),
            dim=256, depth=8, heads=8,
            patch_size=(1,2), window_size=(4,8), flash=True,
        ),
        train=TrainCfg(
            epochs=20,
            learning_rate=1e-4,
            weight_decay=0.01,
            gradient_clip_val=1.0,
            ema_decay=0.999,
            ema_warmup_epochs=5,
            amp=True,
            log_every=50,
            val_every_epochs=1,
            ckpt_dir="./checkpoints",
            run_name="merra2_nextstep",
            resume=None,
        ),
    )

    # ... keep your argparse + args reading ...

    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config, "r") as f:
            user = yaml.safe_load(f) or {}
            # --- allow data.variables as alias for inputs/targets ---
            if "data" in user and isinstance(user["data"], dict):
                d = user["data"]
                if "variables" in d and ("inputs" not in d or "targets" not in d):
                    vars_list = list(d["variables"])
                    d.setdefault("inputs", vars_list)
                    d.setdefault("targets", vars_list)
                    
        # --- robust merge: accepts dicts or dataclass-like objects ---
        def merge(dc, uc):
            # figure out how to iterate 'uc'
            if isinstance(uc, dict):
                items = uc.items()
            else:
                # dataclass / SimpleNamespace / object with attributes
                items = vars(uc).items()

            for k, v in items:
                if not hasattr(dc, k):
                    # unknown key in config; ignore or log if you prefer
                    continue

                dst = getattr(dc, k)

                # case 1: v is a plain dict -> recurse into dst if it's a dataclass-like
                if isinstance(v, dict):
                    if hasattr(dst, "__dict__"):
                        merge(dst, v)
                    else:
                        setattr(dc, k, v)
                    continue

                # case 2: v itself is a dataclass-like object -> recurse
                if hasattr(v, "__dict__"):
                    if hasattr(dst, "__dict__"):
                        merge(dst, v)
                    else:
                        setattr(dc, k, v)
                    continue

                # case 3: leaf (number/str/bool/None/etc.)
                setattr(dc, k, v)

        # build a Cfg object from user yaml (using defaults when missing)
        cfg_from_user = Cfg(
            data=DataCfg(**user.get("data", asdict(default_cfg.data))),
            model=ModelCfg(**user.get("model", asdict(default_cfg.model))),
            train=TrainCfg(**user.get("training", asdict(default_cfg.train))),
        )
        # shallow/deep merge into the defaults
        merge(default_cfg, cfg_from_user)

    cfg = default_cfg
    
    # Expand the semicolon-joined groups from the YAML
    inputs_tokens  = _expand_semicolon_groups(cfg.data.inputs)
    targets_tokens = _expand_semicolon_groups(cfg.data.targets)

    if len(inputs_tokens) == 0 or len(targets_tokens) == 0:
        raise ValueError("Empty inputs/targets after expansion. Check refined_config.yaml.")

    # Optional: log to be sure
    if is_main_process():
        print(f"[cfg] expanded inputs={len(inputs_tokens)} targets={len(targets_tokens)}")

    set_seed(42)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- datasets ---
    train_ds, train_dl = make_loader(
        cfg.data.root_dir, inputs_tokens, "train",
        cfg.data.batch_size, cfg.data.num_workers, cfg.data.pin_memory
    )
    val_ds,   val_dl   = make_loader(
        cfg.data.root_dir, inputs_tokens, "val",  # NOTE: keep same list; ds returns (xb,tb) for SAME vars
        cfg.data.batch_size, cfg.data.num_workers, cfg.data.pin_memory
    )
    # Keep a handle to the sampler to set epoch later
    train_sampler = getattr(train_dl, "sampler", None)

    # NOTE: Our MERRA2SR returns (xb, tb) with the SAME token list it was initialized with.
    # So to train xb->tb on the SAME channel set, just use cfg.data.inputs for BOTH loaders.
    # If you want to keep PHIS input-only (not predicted), set:
    #   inputs  = [..., "surface_geopotential"]
    #   targets = [same list WITHOUT "surface_geopotential"]
    # And build val_ds with that target list (and adjust model out_channels below).

    in_channels  = len(inputs_tokens)
    out_channels = len(targets_tokens)

    # --- model ---
    model = ForecastModel(in_channels=in_channels,
                          out_channels=out_channels,
                          img_resolution=cfg.model.img_resolution,
                          dim=cfg.model.dim,
                          depth=cfg.model.depth,
                          heads=cfg.model.heads,
                          patch_size=cfg.model.patch_size,
                          window_size=cfg.model.window_size,
                          flash=cfg.model.flash)
    model.to(device)
    if ddp_is_active():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if is_main_process():
        print(f"Model params: {count_params(model)/1e6:.2f}M | in={in_channels} out={out_channels}")

    # --- optimizer & scaler ---
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)

    # --- EMA (optional) ---
    use_ema = cfg.train.ema_decay is not None and cfg.train.ema_decay > 0
    ema = EMA(model, decay=cfg.train.ema_decay) if use_ema else None

    # --- resume (optional) ---
    start_epoch, global_step = 0, 0
    if cfg.train.resume and os.path.isfile(cfg.train.resume):
        start_epoch, global_step, loaded_ema, loaded_cfg = load_ckpt(cfg.train.resume, model, opt, scaler)
        if loaded_ema and ema:
            ema.shadow = loaded_ema.shadow
        print(f"Resumed from {cfg.train.resume} @ epoch {start_epoch}, step {global_step}")

    # --- training loop ---
    best_val = math.inf
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    run_tag = cfg.train.run_name

    for epoch in range(start_epoch, cfg.train.epochs):
        if ddp_is_active() and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running = []

        for it, (xb, tb) in enumerate(train_dl, 1):
            xb = xb.to(device, non_blocking=True)  # [B, Cin, H, W]
            tb = tb.to(device, non_blocking=True)  # [B, Cout, H, W]

            Hc, Wc = cfg.model.img_resolution
            xb = center_crop_hw(xb, Hc, Wc)
            tb = center_crop_hw(tb, Hc, Wc)

            # ---- Normalization debug (prints only a few times) ----
            do_log = False
            if getattr(cfg.train, "log_norm", True):
                first_n = int(getattr(cfg.train, "log_norm_first_n", 3))
                every_k = int(getattr(cfg.train, "log_norm_every", 0))  # 0 = disabled
                if it <= first_n:
                    do_log = True
                if every_k > 0 and (it % every_k == 0):
                    do_log = True

            if do_log and is_main_process():
                with torch.no_grad():
                    xb_mean = xb.mean().item()
                    xb_std  = xb.std().item()
                    tb_mean = tb.mean().item()
                    tb_std  = tb.std().item()
                    ch_mean = xb.mean(dim=[0, 2, 3]).detach().cpu()
                    ch_std  = xb.std(dim=[0, 2, 3]).detach().cpu()
                print(
                    f"[norm] ep {epoch} it {it} | "
                    f"xb μ={xb_mean:.4f} σ={xb_std:.4f} | "
                    f"tb μ={tb_mean:.4f} σ={tb_std:.4f} | "
                    f"xb per-ch μ[:4]={ch_mean[:4].tolist()} σ[:4]={ch_std[:4].tolist()}",
                    flush=True
                )
            # -------------------------------------------------------

            if (global_step % cfg.train.accum_steps) == 0:
                opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                y = model(xb)

            # Guard: if model output is non-finite, skip this batch (don’t backprop)
            if not torch.isfinite(y).all():
                print(f"[WARN] non-finite model output at ep {epoch} it {it} "
                    f"(xb μ={xb.mean().item():.3f} σ={xb.std().item():.3f}); skipping batch.",
                    flush=True)
                continue

            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                loss = mse_nan_mean(y, tb) / max(cfg.train.accum_steps, 1)

            # Guard: if loss is non-finite, skip update
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at ep {epoch} it {it}; skipping batch.", flush=True)
                continue

            scaler.scale(loss).backward()

            # step only every accum_steps
            if ((global_step + 1) % cfg.train.accum_steps) == 0:
                if cfg.train.gradient_clip_val and cfg.train.gradient_clip_val > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_val)
                scaler.step(opt)
                scaler.update()

            # EMA update after warmup epochs
            if ema and (epoch >= cfg.train.ema_warmup_epochs):
                ema.update(model)

            running.append(loss.item())
            global_step += 1
            if (it + 1) % cfg.train.log_every == 0 and is_main_process():
                print(f"epoch {epoch+1}/{cfg.train.epochs} | it {it+1}/{len(train_dl)} | "
                    f"loss {sum(running)/len(running):.4f}")

        # end epoch
        dt = time.time() - t0
        train_loss = float(sum(running) / max(len(running), 1))
        if is_main_process():
            print(f"[E{epoch+1}] train_loss={train_loss:.4f} ({dt:.1f}s)")

        if (epoch + 1) % cfg.train.val_every_epochs == 0:
            # validate on EMA weights if available
            if ema:
                # swap in ema weights
                backup = {k: v.clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                val_loss = run_val(model, val_dl, device, cfg.model.img_resolution)
                # restore
                model.load_state_dict(backup, strict=True)
            else:
                val_loss = run_val(model, val_dl, device, cfg.model.img_resolution)

            if is_main_process():
                print(f"[E{epoch+1}] val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    to_save = model.module if ddp_is_active() else model
                    ckpt_path = os.path.join(cfg.train.ckpt_dir, f"{run_tag}_best.pt")
                    save_ckpt(ckpt_path, to_save, opt, scaler, epoch+1, global_step, ema, cfg)
                    print(f"  ↳ saved BEST to {ckpt_path}")

        # periodic "last" checkpoint
        to_save = model.module if ddp_is_active() else model
        ckpt_path = os.path.join(cfg.train.ckpt_dir, f"{run_tag}_last.pt")
        save_ckpt(ckpt_path, to_save, opt, scaler, epoch+1, global_step, ema, cfg)

    print("Training done.")

if __name__ == "__main__":
    main()
