# baseline_lstm_mnl.py
# Baseline-2: LSTM trajectory + MNL (linear) maneuver decision head.

import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio


# -------------------------
# Paths (edit if needed)
# -------------------------
TRAIN_MAT = "./data/TrainSetT.mat"
VAL_MAT   = "./data/ValSetT.mat"
TEST_MAT  = "./data/TestSetT.mat"
CKPT_OUT  = "./baseline_lstm_mnl_best.pt"
COEFF_CSV = "./mnl_coefficients.csv"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset (ego + phi + choice)
# -------------------------
class EgoChoiceDataset(Dataset):
    """
    Returns:
      ego_hist: (H,2) float32 (ego-centric if normalize=True)
      ego_fut : (F,2) float32
      origin  : (2,)  float32
      phi     : (8,)  float32 interpretable state features
      choice  : ()    int64 maneuver class (assumed last column in track)
    """

    def __init__(self, mat_path: str, normalize: bool = True):
        m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        self.traj   = m["traj"]
        self.tracks = m["tracks"]
        self.H      = int(m["historical_length"])
        self.F      = int(m["future_length"])
        self.num_m  = int(m.get("number_of_maneuvers", 9))
        self.normalize = normalize

    def __len__(self) -> int:
        return int(self.traj.shape[0])

    def _get_track(self, ds: int, veh: int):
        ds0, veh0 = ds - 1, veh - 1
        if ds0 < 0 or veh0 < 0:
            return None
        if ds0 >= self.tracks.shape[0] or veh0 >= self.tracks.shape[1]:
            return None
        arr = self.tracks[ds0, veh0]
        if arr is None or getattr(arr, "size", 0) == 0:
            return None
        return arr

    def _slice_at(self, track: np.ndarray, frame: int):
        frames = track[:, 0].astype(int)
        idx = np.where(frames == frame)[0]
        if len(idx) == 0:
            return None

        i0 = int(idx[0])
        lb_h, ub_h = i0 - self.H + 1, i0 + 1
        lb_f, ub_f = i0 + 1, i0 + 1 + self.F
        if lb_h < 0 or ub_f > len(frames):
            return None

        hist = track[lb_h:ub_h, 1:3]
        fut  = track[lb_f:ub_f, 1:3]
        y = int(track[i0, -1])
        y = max(0, min(self.num_m - 1, y))
        return hist, fut, y

    @staticmethod
    def phi_from_hist(hist: np.ndarray) -> np.ndarray:
        H = hist.shape[0]
        d = np.diff(hist, axis=0)

        vx_last = vy_last = 0.0
        if H >= 2:
            vx_last, vy_last = d[-1]

        vx_mean = vy_mean = spd_mean = 0.0
        if H >= 2:
            vx_mean, vy_mean = d.mean(axis=0)
            spd_mean = np.linalg.norm(d, axis=1).mean()

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
            theta = math.atan2(sinang, cosang)

        return np.array(
            [vx_last, vy_last, vx_mean, vy_mean, spd_mean, ax_last, ay_last, theta],
            dtype=np.float32,
        )

    def __getitem__(self, i: int):
        N = len(self)
        for _ in range(50):
            row = self.traj[i]
            ds, veh, fr = int(row[0]), int(row[1]), int(row[2])

            tr = self._get_track(ds, veh)
            if tr is None:
                i = (i + 1) % N
                continue

            sliced = self._slice_at(tr, fr)
            if sliced is None:
                i = (i + 1) % N
                continue

            hist, fut, y = sliced

            if self.normalize:
                origin = hist[-1].astype(np.float32).copy()
                hist = (hist - origin).astype(np.float32)
                fut  = (fut  - origin).astype(np.float32)
            else:
                origin = np.zeros(2, dtype=np.float32)
                hist = hist.astype(np.float32)
                fut  = fut.astype(np.float32)

            phi = self.phi_from_hist(hist)

            return {
                "ego_hist": torch.from_numpy(hist),
                "ego_fut":  torch.from_numpy(fut),
                "origin":   torch.from_numpy(origin),
                "phi":      torch.from_numpy(phi),
                "choice":   torch.tensor(int(y), dtype=torch.long),
            }

        raise RuntimeError("Too many invalid samples encountered while indexing dataset.")


# -------------------------
# Model: shared LSTM + traj head + MNL head
# -------------------------
class LSTM_MNL(nn.Module):
    def __init__(self, hist: int, fut: int, num_classes: int, in_dim: int = 2, hid: int = 128, phi_dim: int = 8):
        super().__init__()
        self.H = hist
        self.F = fut
        self.C = num_classes
        self.hid = hid
        self.phi_dim = phi_dim

        self.enc = nn.LSTM(input_size=in_dim, hidden_size=hid, batch_first=True)

        self.traj_head = nn.Sequential(
            nn.Linear(hid, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fut * 2),
        )

        # MNL-like linear logit layer on [phi, h]
        self.choice_head = nn.Linear(hid + phi_dim, num_classes, bias=True)

        self.feature_names = [
            "vx_last", "vy_last", "vx_mean", "vy_mean",
            "speed_mean", "ax_last", "ay_last", "turn_theta",
        ]

    def forward(self, ego_hist: torch.Tensor, phi: torch.Tensor):
        B = ego_hist.size(0)
        _, (h_n, _) = self.enc(ego_hist)
        h = h_n.squeeze(0)  # (B,hid)

        traj = self.traj_head(h).view(B, self.F, 2)

        z = torch.cat([phi, h], dim=-1)
        logits = self.choice_head(z)
        probs = torch.softmax(logits, dim=-1)
        return traj, logits, probs

    def export_mnl_coeffs(self, out_csv_path: str) -> None:
        W = self.choice_head.weight.detach().cpu().numpy()  # (C, phi+hid)
        b = self.choice_head.bias.detach().cpu().numpy()    # (C,)
        Dphi = len(self.feature_names)

        with open(out_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["class", "ASC"] + self.feature_names + [f"h_{i}" for i in range(W.shape[1] - Dphi)]
            w.writerow(header)
            for c in range(W.shape[0]):
                w.writerow([f"class{c}", float(b[c])] + W[c, :].tolist())


# -------------------------
# Loss
# -------------------------
def ade_fde(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    diff = pred - gt
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)  # (B,F)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


def ade_fde_loss(pred: torch.Tensor, gt: torch.Tensor, lam: float = 1.0):
    ade, fde = ade_fde(pred, gt)
    return ade + lam * fde, ade, fde


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device, lam: float = 1.0, alpha: float = 1.0):
    net.eval()
    Ls, Ltrs, As, Fs, Ces, Accs = [], [], [], [], [], []
    for b in loader:
        hist = b["ego_hist"].to(device)
        fut  = b["ego_fut"].to(device)
        phi  = b["phi"].to(device)
        y    = b["choice"].to(device)

        traj, logits, _ = net(hist, phi)

        Ltr, ADE, FDE = ade_fde_loss(traj, fut, lam=lam)
        CE = F.cross_entropy(logits, y, reduction="mean")
        L  = Ltr + alpha * CE

        pred_cls = logits.argmax(dim=-1)
        acc = (pred_cls == y).float().mean()

        Ls.append(L.item()); Ltrs.append(Ltr.item())
        As.append(ADE.item()); Fs.append(FDE.item())
        Ces.append(CE.item()); Accs.append(acc.item())

    return {
        "loss": float(np.mean(Ls)),
        "traj_loss": float(np.mean(Ltrs)),
        "ADE": float(np.mean(As)),
        "FDE": float(np.mean(Fs)),
        "CE": float(np.mean(Ces)),
        "acc": float(np.mean(Accs)),
    }


# -------------------------
# Train
# -------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EgoChoiceDataset(TRAIN_MAT, normalize=True)
    val_ds   = EgoChoiceDataset(VAL_MAT,   normalize=True)
    test_ds  = EgoChoiceDataset(TEST_MAT,  normalize=True)

    batch_size = 128
    num_workers = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = LSTM_MNL(hist=train_ds.H, fut=train_ds.F, num_classes=train_ds.num_m, in_dim=2, hid=128, phi_dim=8).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    lam = 1.0
    alpha = 1.0
    best_val = float("inf")

    epochs = 20
    for ep in range(1, epochs + 1):
        net.train()
        trL, trA, trF, trCE, trAcc = [], [], [], [], []

        for b in train_loader:
            hist = b["ego_hist"].to(device)
            fut  = b["ego_fut"].to(device)
            phi  = b["phi"].to(device)
            y    = b["choice"].to(device)

            traj, logits, _ = net(hist, phi)

            Ltr, ADE, FDE = ade_fde_loss(traj, fut, lam=lam)
            CE = F.cross_entropy(logits, y, reduction="mean")
            loss = Ltr + alpha * CE

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            acc = (logits.argmax(-1) == y).float().mean()

            trL.append(loss.item()); trA.append(ADE.item()); trF.append(FDE.item())
            trCE.append(CE.item()); trAcc.append(acc.item())

        val = evaluate(net, val_loader, device, lam=lam, alpha=alpha)

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(net.state_dict(), CKPT_OUT)

        print(
            f"Epoch {ep:02d} | "
            f"train L {np.mean(trL):.4f} (ADE {np.mean(trA):.4f}, FDE {np.mean(trF):.4f}, CE {np.mean(trCE):.4f}, acc {np.mean(trAcc):.3f}) | "
            f"val L {val['loss']:.4f} (ADE {val['ADE']:.4f}, FDE {val['FDE']:.4f}, CE {val['CE']:.4f}, acc {val['acc']:.3f})"
            f"{'  [best]' if val['loss'] == best_val else ''}"
        )

    net.load_state_dict(torch.load(CKPT_OUT, map_location=device))
    test = evaluate(net, test_loader, device, lam=lam, alpha=alpha)
    print(f"\nTEST | L {test['loss']:.4f} | ADE {test['ADE']:.4f} | FDE {test['FDE']:.4f} | CE {test['CE']:.4f} | acc {test['acc']:.3f}")
    print(f"Saved: {CKPT_OUT}")

    net.export_mnl_coeffs(COEFF_CSV)
    print(f"Exported MNL coefficients: {COEFF_CSV}")


if __name__ == "__main__":
    main()
