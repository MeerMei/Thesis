# Clean evaluation for 3 HighD trajectory models:
#   1) Baseline-1: LSTM-only (trajectory)
#   2) Baseline-2: LSTM+MNL (trajectory + maneuver probs)
#   3) Proposed : LSTM+MNL+VAE (trajectory + maneuver probs + latent z)
#
# Computes:
#   - Trajectory: ADE, FDE, per-horizon ADE curve
#   - Choice: Accuracy, Macro-F1, Cross-Entropy, Confusion matrix
#   - Calibration: reliability diagram + ECE
#   - Uncertainty: paired bootstrap CIs for metric differences
# Exports:
#   - Excel workbook (metrics + bootstrap CIs + confusion matrices + coefficients table)
#   - Figures (histograms, ADE curve, confusion, reliability)
#
# Usage (example):
#   python eval_three_models.py ^
#     --test_mat ./data/TestSetT.mat ^
#     --ckpt_lstm ./baseline_lstm_best.pt ^
#     --ckpt_mnl  ./baseline_lstm_mnl_best.pt ^
#     --ckpt_vae  ./proposed_lstm_mnl_vae_best.pt ^
#     --outdir ./eval_out ^
#     --device auto
#
# Notes:
#   - This script assumes your training files are importable in the same folder:
#       baseline_lstm_full.py (BaselineLSTM)
#       baseline_lstm_mnl.py  (LSTM_MNL)
#       proposed_lstm_mnl_vae.py (ProposedLSTM_MNL_VAE)
#   - The .mat file is the output from your MATLAB HighD_preprocess.

from __future__ import annotations

import os
import math
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_savefig(path: str, dpi: int = 150) -> None:
    """Save figure but don't crash if disk is full."""
    try:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[FIG] {path}")
    except OSError as e:
        print(f"[WARN] Could not save {path}: {e}")
    finally:
        plt.close()

def load_mat(mat_path: str) -> Dict:
    """Load MATLAB v7 .mat; if v7.3 use h5py (not handled here)."""
    try:
        return sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:
        raise RuntimeError(
            f"Could not load {mat_path}. It might be MATLAB v7.3 (HDF5). "
            f"Convert to v7 or use h5py. Original error: {e}"
        )

def find_frame_index(frames_sorted: np.ndarray, frame_id: int) -> int:
    j = np.searchsorted(frames_sorted, frame_id)
    if j < len(frames_sorted) and int(frames_sorted[j]) == int(frame_id):
        return int(j)
    return -1


# =========================
# Dataset
# =========================
class HighDEvalDataset(Dataset):
    """
    Provides the minimum inputs for all three models:
      ego_hist : (H,2) ego-centric history
      ego_fut  : (F,2) ego-centric future (ground truth)
      origin   : (2,) absolute last-history point (for de-normalization)
      phi      : (8,) interpretable feature vector for choice head
      choice   : () maneuver class id in [0..C-1]
    Assumes: per-track matrix columns:
      [frame, x, y, ..., class_id_last_col]
    """
    def __init__(self, mat_path: str, normalize: bool = True):
        m = load_mat(mat_path)
        self.traj   = m["traj"]
        self.tracks = m["tracks"]
        self.H      = int(m["historical_length"])
        self.F      = int(m["future_length"])
        self.C      = int(m.get("number_of_maneuvers", 9))
        self.normalize = normalize

    def __len__(self) -> int:
        return int(self.traj.shape[0])

    def _get_track(self, ds: int, veh: int) -> Optional[np.ndarray]:
        ds0, veh0 = ds - 1, veh - 1  # MATLAB -> Python indexing
        if ds0 < 0 or veh0 < 0:
            return None
        if ds0 >= self.tracks.shape[0] or veh0 >= self.tracks.shape[1]:
            return None
        arr = self.tracks[ds0, veh0]
        if arr is None or getattr(arr, "size", 0) == 0:
            return None
        return arr

    def _slice(self, tr: np.ndarray, frame: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        frames = tr[:, 0].astype(int)
        idx = np.where(frames == frame)[0]
        if len(idx) == 0:
            return None, None, -1
        i0 = int(idx[0])
        lb_h, ub_h = i0 - self.H + 1, i0 + 1
        lb_f, ub_f = i0 + 1, i0 + 1 + self.F
        if lb_h < 0 or ub_f > len(frames):
            return None, None, -1
        hist = tr[lb_h:ub_h, 1:3].astype(np.float32)   # (H,2)
        fut  = tr[lb_f:ub_f, 1:3].astype(np.float32)   # (F,2)
        y = int(tr[i0, -1])
        if y < 0 or y >= self.C:
            y = max(0, min(self.C - 1, y))
        return hist, fut, y

    @staticmethod
    def phi_from_hist(hist: np.ndarray) -> np.ndarray:
        """
        φ(x_t) from ego-centric history:
          [vx_last, vy_last, vx_mean, vy_mean, speed_mean, ax_last, ay_last, turn_theta]
        """
        H = hist.shape[0]
        d = np.diff(hist, axis=0)  # (H-1,2)
        vx_last = vy_last = 0.0
        if H >= 2:
            vx_last, vy_last = d[-1]
        vx_mean = vy_mean = spd_mean = 0.0
        if H >= 2:
            vx_mean, vy_mean = d.mean(axis=0)
            spd_mean = float(np.linalg.norm(d, axis=1).mean())
        ax_last = ay_last = 0.0
        if H >= 3:
            a = np.diff(d, axis=0)
            ax_last, ay_last = a[-1]
        theta = 0.0
        if H >= 3:
            u, v = d[-2], d[-1]
            nu = np.linalg.norm(u) + 1e-6
            nv = np.linalg.norm(v) + 1e-6
            cosang = np.clip((u @ v) / (nu * nv), -1.0, 1.0)
            sinang = (u[0] * v[1] - u[1] * v[0]) / (nu * nv)
            theta = float(math.atan2(sinang, cosang))
        return np.array([vx_last, vy_last, vx_mean, vy_mean, spd_mean, ax_last, ay_last, theta], dtype=np.float32)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.traj[i]
        ds, veh, fr = int(row[0]), int(row[1]), int(row[2])
        tr = self._get_track(ds, veh)
        if tr is None:
            return self.__getitem__((i + 1) % len(self))
        hist, fut, y = self._slice(tr, fr)
        if hist is None:
            return self.__getitem__((i + 1) % len(self))

        if self.normalize:
            origin = hist[-1].copy()
            hist = hist - origin
            fut  = fut  - origin
        else:
            origin = np.zeros(2, dtype=np.float32)

        phi = self.phi_from_hist(hist)
        return {
            "ego_hist": torch.from_numpy(hist),                  # (H,2)
            "ego_fut":  torch.from_numpy(fut),                   # (F,2)
            "origin":   torch.from_numpy(origin),                # (2,)
            "phi":      torch.from_numpy(phi),                   # (8,)
            "choice":   torch.tensor(int(y), dtype=torch.long),   # ()
        }


# =========================
# Metrics (trajectory)
# =========================
def per_sample_ade_fde(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pred, gt: (N,F,2)
    returns:
      ade: (N,)
      fde: (N,)
      ade_curve: (F,) mean error at each horizon step
    """
    diff = pred - gt
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-6)  # (N,F)
    ade = dist.mean(axis=1)
    fde = dist[:, -1]
    ade_curve = dist.mean(axis=0)
    return ade, fde, ade_curve


# =========================
# Metrics (choice & calibration)
# =========================
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size else float("nan")

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> float:
    """Macro F1 without sklearn."""
    f1s = []
    for c in range(C):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")

def cross_entropy_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    p = probs[np.arange(probs.shape[0]), y_true]
    p = np.clip(p, 1e-12, 1.0)
    return float((-np.log(p)).mean())

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < C and 0 <= p < C:
            cm[t, p] += 1
    return cm

def ece_multiclass(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error (ECE) for multi-class:
    use confidence = max prob; correctness = (argmax == y_true).
    Returns:
      ece, bin_edges, acc_per_bin, conf_per_bin
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    accs = np.zeros(n_bins, dtype=np.float64)
    confs = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    bin_ids = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        m = bin_ids == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            accs[b] = float(correct[m].mean())
            confs[b] = float(conf[m].mean())

    ece = 0.0
    N = max(1, len(conf))
    for b in range(n_bins):
        if counts[b] > 0:
            ece += (counts[b] / N) * abs(accs[b] - confs[b])

    return float(ece), edges, accs, confs


# =========================
# Bootstrap (paired)
# =========================
def paired_bootstrap_ci(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_boot: int,
    ci: float,
    seed: int
) -> Dict[str, float]:
    """
    Paired bootstrap CI for mean difference (B - A).
    x_a, x_b: per-sample metric arrays (same length).
    returns: mean_diff, ci_low, ci_high
    """
    n = min(len(x_a), len(x_b))
    x_a = x_a[:n]
    x_b = x_b[:n]
    d = x_b - x_a
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(d[idx].mean())
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return {"mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi}


# =========================
# Plotting
# =========================
def plot_hist_step_overlay(a: np.ndarray, b: np.ndarray, bins: int, xlabel: str, title: str, out_png: str,
                           label_a: str = "A", label_b: str = "B") -> None:
    """
    Step hist overlay (clearer than overlapping bars).
    Uses density=True for comparability.
    """
    plt.figure(figsize=(6, 4))
    ha, edges = np.histogram(a, bins=bins, density=True)
    hb, _     = np.histogram(b, bins=edges, density=True)

    # Step plot: append last value to match edges length (avoids extra diagonal line)
    plt.step(edges, np.r_[ha, ha[-1]], where="post", lw=2, label=label_a)
    plt.step(edges, np.r_[hb, hb[-1]], where="post", lw=2, ls="--", label=label_b)

    plt.xlabel(xlabel); plt.ylabel("density"); plt.title(title)
    plt.grid(True, lw=0.3); plt.legend()
    safe_savefig(out_png)

def plot_ade_curve(curves: Dict[str, np.ndarray], out_png: str) -> None:
    plt.figure(figsize=(6, 4))
    for name, curve in curves.items():
        plt.plot(np.arange(curve.shape[0]), curve, lw=2, label=name)
    plt.xlabel("Horizon step k")
    plt.ylabel("mean ADE at step k")
    plt.title("Per-horizon ADE")
    plt.grid(True, lw=0.3)
    plt.legend()
    safe_savefig(out_png)

def plot_confusion(cm: np.ndarray, out_png: str, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    safe_savefig(out_png)

def plot_reliability(edges: np.ndarray, accs: np.ndarray, confs: np.ndarray, out_png: str, title: str) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], ls="--", lw=1, label="perfect")
    plt.plot(centers, accs, marker="o", lw=2, label="accuracy")
    plt.plot(centers, confs, marker="s", lw=2, label="confidence")
    plt.xlabel("confidence bin")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, lw=0.3)
    plt.legend()
    safe_savefig(out_png)


# =========================
# Model loading
# =========================
def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def load_model_strict(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

def infer_proposed_dims_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Robustly infer z_dim and expert input dim from checkpoint weights to avoid size mismatch.
    Expects keys:
      mu_head.weight : (z_dim, hid)
      logsig_head.weight : (z_dim, hid)
      experts.0.0.weight : (hidden, hid+z_dim)  # first Linear in expert MLP
    """
    z_dim = int(state["mu_head.weight"].shape[0]) if "mu_head.weight" in state else 8
    hid   = int(state["mu_head.weight"].shape[1]) if "mu_head.weight" in state else 128
    return {"z_dim": z_dim, "hid": hid}

def load_proposed_model(device: torch.device, ckpt_path: str, H: int, F: int, C: int) -> nn.Module:
    """
    Imports ProposedLSTM_MNL_VAE and constructs with checkpoint-consistent z_dim if possible.
    """
    try:
        from proposed_lstm_mnl_vae import ProposedLSTM_MNL_VAE
    except Exception as e:
        raise RuntimeError("Could not import ProposedLSTM_MNL_VAE from proposed_lstm_mnl_vae.py") from e

    state = torch.load(ckpt_path, map_location="cpu")
    dims = infer_proposed_dims_from_state(state)
    z_dim = dims["z_dim"]
    hid   = dims["hid"]

    # Instantiate. Your implementation should accept z_dim; if not, adapt your class accordingly.
    model = ProposedLSTM_MNL_VAE(hist=H, fut=F, in_dim=2, hid=hid, phi_dim=8, num_classes=C, z_dim=z_dim, moe=True)
    model = load_model_strict(model, ckpt_path, device)
    return model


# =========================
# Evaluation core
# =========================
@dataclass
class ModelOutputs:
    name: str
    pred_abs: np.ndarray     # (N,F,2)
    gt_abs: np.ndarray       # (N,F,2)
    ade: np.ndarray          # (N,)
    fde: np.ndarray          # (N,)
    ade_curve: np.ndarray    # (F,)
    probs: Optional[np.ndarray] = None   # (N,C) if available
    y_true: Optional[np.ndarray] = None  # (N,)
    y_pred: Optional[np.ndarray] = None  # (N,)
    ce: Optional[float] = None
    acc: Optional[float] = None
    f1: Optional[float] = None
    ece: Optional[float] = None
    conf: Optional[np.ndarray] = None
    rel_edges: Optional[np.ndarray] = None
    rel_accs: Optional[np.ndarray] = None
    rel_confs: Optional[np.ndarray] = None


@torch.no_grad()
def run_model_inference(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str,
    max_batches: Optional[int] = None,
    amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    kind:
      - "lstm" : model(hist)->traj
      - "mnl"  : model(hist,phi)->(traj,logits,probs)
      - "vae"  : proposed; expected model(hist,phi)->(traj,logits,probs,extras?) OR (traj,logits,probs)
    Returns:
      pred_abs (N,F,2), gt_abs (N,F,2), probs (N,C or None), y_true (N,), y_pred (N,)
    """
    preds, gts = [], []
    probs_all = []
    y_true_all = []
    y_pred_all = []

    use_amp = (amp and device.type == "cuda")
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        hist = batch["ego_hist"].to(device, non_blocking=True)
        fut  = batch["ego_fut"].to(device, non_blocking=True)
        org  = batch["origin"].to(device, non_blocking=True)
        phi  = batch["phi"].to(device, non_blocking=True)
        y    = batch["choice"].to(device, non_blocking=True)

        if kind == "lstm":
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                traj = model(hist)
            logits = probs = None

        elif kind == "mnl":
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                traj, logits, probs = model(hist, phi)

        elif kind == "vae":
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(hist, phi)
            # support either (traj,logits,probs) or (traj,logits,probs, extras...)
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                traj, logits, probs = out[0], out[1], out[2]
            else:
                raise RuntimeError("Proposed model forward() must return at least (traj, logits, probs).")
        else:
            raise ValueError(f"Unknown kind: {kind}")

        pred_abs = traj + org[:, None, :]
        gt_abs   = fut  + org[:, None, :]

        preds.append(to_np(pred_abs).astype(np.float32))
        gts.append(to_np(gt_abs).astype(np.float32))

        if probs is not None:
            p = to_np(probs).astype(np.float32)
            probs_all.append(p)
            yp = p.argmax(axis=-1).astype(np.int64)
            y_pred_all.append(yp)
            y_true_all.append(to_np(y).astype(np.int64))

    pred_abs = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1, 2), np.float32)
    gt_abs   = np.concatenate(gts, axis=0) if gts else np.zeros((0, 1, 2), np.float32)

    if probs_all:
        probs_np = np.concatenate(probs_all, axis=0)
        y_true_np = np.concatenate(y_true_all, axis=0)
        y_pred_np = np.concatenate(y_pred_all, axis=0)
    else:
        probs_np = y_true_np = y_pred_np = None

    return pred_abs, gt_abs, probs_np, y_true_np, y_pred_np


def compute_outputs(
    name: str,
    pred_abs: np.ndarray,
    gt_abs: np.ndarray,
    probs: Optional[np.ndarray],
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    C: int,
    n_bins: int
) -> ModelOutputs:
    ade, fde, ade_curve = per_sample_ade_fde(pred_abs, gt_abs)

    out = ModelOutputs(
        name=name,
        pred_abs=pred_abs,
        gt_abs=gt_abs,
        ade=ade,
        fde=fde,
        ade_curve=ade_curve,
        probs=probs,
        y_true=y_true,
        y_pred=y_pred
    )

    if probs is not None and y_true is not None and y_pred is not None:
        out.acc = accuracy(y_true, y_pred)
        out.f1  = macro_f1(y_true, y_pred, C=C)
        out.ce  = cross_entropy_from_probs(probs, y_true)
        out.conf = confusion_matrix(y_true, y_pred, C=C)
        ece, edges, accs, confs = ece_multiclass(probs, y_true, n_bins=n_bins)
        out.ece = ece
        out.rel_edges = edges
        out.rel_accs = accs
        out.rel_confs = confs

    return out


# =========================
# Excel export
# =========================
def export_excel(
    outputs: List[ModelOutputs],
    bootstrap: List[Dict],
    out_xlsx: str,
    coeff_tables: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    rows = []
    for o in outputs:
        rows.append({
            "model": o.name,
            "ADE_mean": float(np.mean(o.ade)),
            "ADE_median": float(np.median(o.ade)),
            "FDE_mean": float(np.mean(o.fde)),
            "FDE_median": float(np.median(o.fde)),
            "Acc": o.acc,
            "MacroF1": o.f1,
            "CE": o.ce,
            "ECE": o.ece,
            "N": int(o.ade.shape[0]),
            "F": int(o.ade_curve.shape[0]),
        })
    df_metrics = pd.DataFrame(rows)

    df_boot = pd.DataFrame(bootstrap) if bootstrap else pd.DataFrame()

    # Confusions
    conf_sheets = {}
    for o in outputs:
        if o.conf is not None:
            conf_sheets[o.name] = pd.DataFrame(o.conf)

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_metrics.to_excel(w, sheet_name="Metrics", index=False)
            if not df_boot.empty:
                df_boot.to_excel(w, sheet_name="Bootstrap", index=False)

            for name, df in conf_sheets.items():
                sheet = f"Conf_{name}"[:31]
                df.to_excel(w, sheet_name=sheet, index=True)

            if coeff_tables:
                for name, df in coeff_tables.items():
                    sheet = f"Coeff_{name}"[:31]
                    df.to_excel(w, sheet_name=sheet, index=False)

        print(f"[XLSX] {out_xlsx}")
    except OSError as e:
        print(f"[WARN] Could not write Excel {out_xlsx}: {e}")


# =========================
# Coefficients export (interpretability)
# =========================
def extract_choice_coeffs_mnl(model: nn.Module, feature_names: List[str]) -> pd.DataFrame:
    """
    For Baseline-2 LSTM+MNL:
      choice_head: Linear([phi,h]) -> logits
    Returns a table for the phi block only:
      columns: class, ASC, beta_<feature>
    """
    if not hasattr(model, "choice_head"):
        raise ValueError("Model has no choice_head.")
    layer = model.choice_head
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy()
    Dphi = len(feature_names)
    rows = []
    for c in range(W.shape[0]):
        row = {"class": c, "ASC": float(b[c])}
        for j, fn in enumerate(feature_names):
            row[f"beta_{fn}"] = float(W[c, j])  # first block is phi by construction in our training code
        rows.append(row)
    return pd.DataFrame(rows)

def extract_choice_coeffs_proposed(model: nn.Module, feature_names: List[str]) -> pd.DataFrame:
    """
    For Proposed model, prefer a direct phi->logit linear layer if present.
    Supported patterns:
      - model.W_phi : Linear(phi)->logits
      - model.beta_phi : Linear(phi)->logits
      - otherwise falls back to model.choice_head (assumed concat [phi,h,z])
    """
    layer = None
    for attr in ["W_phi", "beta_phi", "phi_head"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            break
    if layer is None and hasattr(model, "choice_head"):
        layer = getattr(model, "choice_head")

    if layer is None:
        raise ValueError("No recognizable phi->logit layer found.")

    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)

    rows = []
    Dphi = len(feature_names)
    for c in range(W.shape[0]):
        row = {"class": c, "ASC": float(b[c])}
        for j, fn in enumerate(feature_names):
            if j < W.shape[1]:
                row[f"beta_{fn}"] = float(W[c, j])
        rows.append(row)
    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_mat", required=True, help="Path to TestSetT.mat")
    p.add_argument("--ckpt_lstm", required=True, help="baseline_lstm_best.pt")
    p.add_argument("--ckpt_mnl",  required=True, help="baseline_lstm_mnl_best.pt")
    p.add_argument("--ckpt_vae",  required=True, help="proposed_lstm_mnl_vae_best.pt")
    p.add_argument("--outdir", default="./eval_out", help="Output directory")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_batches", type=int, default=0, help="0 = all batches; >0 to cap")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP on CUDA")
    p.add_argument("--ece_bins", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap_ci", type=float, default=0.95)
    p.add_argument("--bootstrap_seed", type=int, default=7)
    p.add_argument("--bins_hist", type=int, default=80)
    args = p.parse_args()

    safe_makedirs(args.outdir)
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[SETUP] device={device} | batch={args.batch} | workers={args.num_workers}")

    # Data
    ds = HighDEvalDataset(args.test_mat, normalize=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    max_batches = None if args.max_batches <= 0 else args.max_batches

    # Import and load models
    try:
        from baseline_lstm_full import BaselineLSTM
    except Exception as e:
        raise RuntimeError("Could not import BaselineLSTM from baseline_lstm_full.py") from e

    try:
        from baseline_lstm_mnl import LSTM_MNL
    except Exception as e:
        raise RuntimeError("Could not import LSTM_MNL from baseline_lstm_mnl.py") from e

    lstm = BaselineLSTM(hist=ds.H, fut=ds.F, in_dim=2, hid=128, mode="mlp")
    lstm = load_model_strict(lstm, args.ckpt_lstm, device)

    mnl = LSTM_MNL(hist=ds.H, fut=ds.F, in_dim=2, hid=128, phi_dim=8, num_classes=ds.C, mode="mlp")
    mnl = load_model_strict(mnl, args.ckpt_mnl, device)

    vae = load_proposed_model(device, args.ckpt_vae, H=ds.H, F=ds.F, C=ds.C)

    # Inference
    amp = (not args.no_amp)
    print("[RUN] Inference LSTM...")
    pr_l, gt_l, _, _, _ = run_model_inference("LSTM", lstm, loader, device, kind="lstm", max_batches=max_batches, amp=amp)
    out_l = compute_outputs("LSTM", pr_l, gt_l, None, None, None, C=ds.C, n_bins=args.ece_bins)

    print("[RUN] Inference LSTM+MNL...")
    pr_m, gt_m, probs_m, y_t, y_p = run_model_inference("MNL", mnl, loader, device, kind="mnl", max_batches=max_batches, amp=amp)
    out_m = compute_outputs("LSTM+MNL", pr_m, gt_m, probs_m, y_t, y_p, C=ds.C, n_bins=args.ece_bins)

    print("[RUN] Inference Proposed (VAE)...")
    pr_v, gt_v, probs_v, y_tv, y_pv = run_model_inference("VAE", vae, loader, device, kind="vae", max_batches=max_batches, amp=amp)
    out_v = compute_outputs("Proposed", pr_v, gt_v, probs_v, y_tv, y_pv, C=ds.C, n_bins=args.ece_bins)

    outputs = [out_l, out_m, out_v]

    # Console summary
    def summ(o: ModelOutputs) -> str:
        s = f"{o.name:10s} | ADE mean {o.ade.mean():.4f} med {np.median(o.ade):.4f} | FDE mean {o.fde.mean():.4f} med {np.median(o.fde):.4f}"
        if o.acc is not None:
            s += f" | Acc {o.acc:.3f} | F1 {o.f1:.3f} | CE {o.ce:.4f} | ECE {o.ece:.4f}"
        return s

    print("\n=== SUMMARY (Test) ===")
    for o in outputs:
        print(summ(o))

    # Bootstrap CIs for per-sample ADE/FDE deltas
    boot_rows = []
    if args.bootstrap > 0:
        pairs = [
            ("LSTM", "LSTM+MNL", out_l, out_m),
            ("LSTM", "Proposed", out_l, out_v),
            ("LSTM+MNL", "Proposed", out_m, out_v),
        ]
        for metric_name in ["ADE", "FDE"]:
            for a_name, b_name, a_out, b_out in pairs:
                x_a = a_out.ade if metric_name == "ADE" else a_out.fde
                x_b = b_out.ade if metric_name == "ADE" else b_out.fde
                ci = paired_bootstrap_ci(
                    x_a=x_a, x_b=x_b,
                    n_boot=args.bootstrap,
                    ci=args.bootstrap_ci,
                    seed=args.bootstrap_seed + (0 if metric_name == "ADE" else 100)
                )
                boot_rows.append({
                    "metric": metric_name,
                    "A": a_name,
                    "B": b_name,
                    "mean_diff_(B-A)": ci["mean_diff"],
                    f"ci{int(args.bootstrap_ci*100)}_low": ci["ci_low"],
                    f"ci{int(args.bootstrap_ci*100)}_high": ci["ci_high"],
                    "n_boot": args.bootstrap,
                })

        # Classification metrics (paired bootstrap) for MNL vs Proposed only (requires probs)
        if out_m.probs is not None and out_v.probs is not None:
            n = min(out_m.probs.shape[0], out_v.probs.shape[0])
            y_true = out_m.y_true[:n]
            # per-sample negative log likelihood (NLL) arrays for paired bootstrap
            nll_m = -np.log(np.clip(out_m.probs[:n, :][np.arange(n), y_true], 1e-12, 1.0))
            nll_v = -np.log(np.clip(out_v.probs[:n, :][np.arange(n), y_true], 1e-12, 1.0))
            ci_ce = paired_bootstrap_ci(nll_m, nll_v, args.bootstrap, args.bootstrap_ci, args.bootstrap_seed + 200)
            boot_rows.append({
                "metric": "CE(NLL)",
                "A": "LSTM+MNL",
                "B": "Proposed",
                "mean_diff_(B-A)": ci_ce["mean_diff"],
                f"ci{int(args.bootstrap_ci*100)}_low": ci_ce["ci_low"],
                f"ci{int(args.bootstrap_ci*100)}_high": ci_ce["ci_high"],
                "n_boot": args.bootstrap,
            })

    # Figures: hist overlays
    bins = args.bins_hist
    plot_hist_step_overlay(out_l.ade, out_m.ade, bins=bins, xlabel="ADE", title="ADE: LSTM vs LSTM+MNL",
                           out_png=os.path.join(args.outdir, "ade_hist_lstm_vs_mnl.png"),
                           label_a="LSTM", label_b="LSTM+MNL")
    plot_hist_step_overlay(out_l.ade, out_v.ade, bins=bins, xlabel="ADE", title="ADE: LSTM vs Proposed",
                           out_png=os.path.join(args.outdir, "ade_hist_lstm_vs_proposed.png"),
                           label_a="LSTM", label_b="Proposed")
    plot_hist_step_overlay(out_m.ade, out_v.ade, bins=bins, xlabel="ADE", title="ADE: LSTM+MNL vs Proposed",
                           out_png=os.path.join(args.outdir, "ade_hist_mnl_vs_proposed.png"),
                           label_a="LSTM+MNL", label_b="Proposed")

    plot_hist_step_overlay(out_l.fde, out_m.fde, bins=bins, xlabel="FDE", title="FDE: LSTM vs LSTM+MNL",
                           out_png=os.path.join(args.outdir, "fde_hist_lstm_vs_mnl.png"),
                           label_a="LSTM", label_b="LSTM+MNL")
    plot_hist_step_overlay(out_l.fde, out_v.fde, bins=bins, xlabel="FDE", title="FDE: LSTM vs Proposed",
                           out_png=os.path.join(args.outdir, "fde_hist_lstm_vs_proposed.png"),
                           label_a="LSTM", label_b="Proposed")
    plot_hist_step_overlay(out_m.fde, out_v.fde, bins=bins, xlabel="FDE", title="FDE: LSTM+MNL vs Proposed",
                           out_png=os.path.join(args.outdir, "fde_hist_mnl_vs_proposed.png"),
                           label_a="LSTM+MNL", label_b="Proposed")

    # Figure: ADE curve
    plot_ade_curve(
        {"LSTM": out_l.ade_curve, "LSTM+MNL": out_m.ade_curve, "Proposed": out_v.ade_curve},
        out_png=os.path.join(args.outdir, "ade_curve_all.png"),
    )

    # Confusions + reliability
    if out_m.conf is not None:
        plot_confusion(out_m.conf, os.path.join(args.outdir, "confusion_mnl.png"), "Confusion (LSTM+MNL)")
    if out_v.conf is not None:
        plot_confusion(out_v.conf, os.path.join(args.outdir, "confusion_proposed.png"), "Confusion (Proposed)")

    if out_m.rel_edges is not None:
        plot_reliability(out_m.rel_edges, out_m.rel_accs, out_m.rel_confs,
                         os.path.join(args.outdir, "reliability_mnl.png"),
                         title=f"Reliability (LSTM+MNL) | ECE={out_m.ece:.4f}")
    if out_v.rel_edges is not None:
        plot_reliability(out_v.rel_edges, out_v.rel_accs, out_v.rel_confs,
                         os.path.join(args.outdir, "reliability_proposed.png"),
                         title=f"Reliability (Proposed) | ECE={out_v.ece:.4f}")

    # Coefficient tables (interpretability)
    coeff_tables = {}
    feature_names = ["vx_last", "vy_last", "vx_mean", "vy_mean", "speed_mean", "ax_last", "ay_last", "turn_theta"]
    try:
        coeff_tables["MNL"] = extract_choice_coeffs_mnl(mnl, feature_names)
    except Exception as e:
        warnings.warn(f"Could not extract MNL coeffs: {e}")
    try:
        coeff_tables["Proposed"] = extract_choice_coeffs_proposed(vae, feature_names)
    except Exception as e:
        warnings.warn(f"Could not extract Proposed coeffs: {e}")

    # Excel export
    out_xlsx = os.path.join(args.outdir, "three_models_evaluation.xlsx")
    export_excel(outputs, boot_rows, out_xlsx, coeff_tables=coeff_tables)

    # Save run config for reproducibility
    cfg_path = os.path.join(args.outdir, "eval_config.json")
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[CFG] {cfg_path}")
    except OSError as e:
        print(f"[WARN] Could not write {cfg_path}: {e}")

    print("\nDone. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
