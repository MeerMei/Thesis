# baseline_lstm_full.py
# Baseline-1: Ego-only LSTM trajectory prediction on HighD .mat splits.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio


# -------------------------
# Paths (edit if needed)
# -------------------------
TRAIN_MAT = "./data/TrainSetT.mat"
VAL_MAT   = "./data/ValSetT.mat"
TEST_MAT  = "./data/TestSetT.mat"
CKPT_OUT  = "./baseline_lstm_best.pt"


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset (ego-only)
# -------------------------
class EgoOnlyDataset(Dataset):
    """
    Loads HighD preprocessed .mat and returns:
      ego_hist: (H,2) float32  (ego-centric if normalize=True)
      ego_fut : (F,2) float32  (ego-centric if normalize=True)
      origin  : (2,) float32   last history point in absolute coords
    """

    def __init__(self, mat_path: str, normalize: bool = True):
        m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        self.traj   = m["traj"]     # rows: [dsId, vehId, frameId, ...]
        self.tracks = m["tracks"]   # 2D cell-like array: {ds,veh} -> (T,K)

        self.H = int(m["historical_length"])
        self.F = int(m["future_length"])

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

        hist = track[lb_h:ub_h, 1:3]  # (H,2)
        fut  = track[lb_f:ub_f, 1:3]  # (F,2)
        return hist, fut

    def __getitem__(self, i: int):
        # robustly skip bad rows without recursion
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

            hist, fut = sliced

            if self.normalize:
                origin = hist[-1].astype(np.float32).copy()
                hist = (hist - origin).astype(np.float32)
                fut  = (fut  - origin).astype(np.float32)
            else:
                origin = np.zeros(2, dtype=np.float32)
                hist = hist.astype(np.float32)
                fut  = fut.astype(np.float32)

            return {
                "ego_hist": torch.from_numpy(hist),
                "ego_fut":  torch.from_numpy(fut),
                "origin":   torch.from_numpy(origin),
            }

        raise RuntimeError("Too many invalid samples encountered while indexing dataset.")


# -------------------------
# Model
# -------------------------
class BaselineLSTM(nn.Module):
    """
    LSTM encoder over history -> MLP head -> (F,2) future trajectory.
    """

    def __init__(self, hist: int, fut: int, in_dim: int = 2, hid: int = 128):
        super().__init__()
        self.H = hist
        self.F = fut
        self.hid = hid

        self.enc = nn.LSTM(input_size=in_dim, hidden_size=hid, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fut * 2),
        )

    def forward(self, ego_hist: torch.Tensor) -> torch.Tensor:
        B = ego_hist.size(0)
        _, (h_n, _) = self.enc(ego_hist)          # h_n: (1,B,hid)
        h = h_n.squeeze(0)                        # (B,hid)
        out = self.head(h).view(B, self.F, 2)     # (B,F,2)
        return out


# -------------------------
# Metrics / loss
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
def eval_loader(net: nn.Module, loader: DataLoader, device: torch.device, lam: float = 1.0):
    net.eval()
    Ls, As, Fs = [], [], []
    for batch in loader:
        hist = batch["ego_hist"].to(device)
        fut  = batch["ego_fut"].to(device)
        pred = net(hist)
        L, A, F_ = ade_fde_loss(pred, fut, lam=lam)
        Ls.append(L.item()); As.append(A.item()); Fs.append(F_.item())
    return float(np.mean(Ls)), float(np.mean(As)), float(np.mean(Fs))


# -------------------------
# Train
# -------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EgoOnlyDataset(TRAIN_MAT, normalize=True)
    val_ds   = EgoOnlyDataset(VAL_MAT,   normalize=True)
    test_ds  = EgoOnlyDataset(TEST_MAT,  normalize=True)

    batch_size = 128
    num_workers = 2  # keep modest for Windows stability

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = BaselineLSTM(hist=train_ds.H, fut=train_ds.F, in_dim=2, hid=128).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    lam = 1.0
    best_val = float("inf")

    epochs = 20
    for ep in range(1, epochs + 1):
        net.train()
        trL, trA, trF = [], [], []

        for batch in train_loader:
            hist = batch["ego_hist"].to(device)
            fut  = batch["ego_fut"].to(device)

            pred = net(hist)
            loss, ade, fde = ade_fde_loss(pred, fut, lam=lam)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            trL.append(loss.item()); trA.append(ade.item()); trF.append(fde.item())

        valL, valA, valF = eval_loader(net, val_loader, device, lam=lam)

        if valL < best_val:
            best_val = valL
            torch.save(net.state_dict(), CKPT_OUT)

        print(
            f"Epoch {ep:02d} | "
            f"train L {np.mean(trL):.4f} (ADE {np.mean(trA):.4f}, FDE {np.mean(trF):.4f}) | "
            f"val L {valL:.4f} (ADE {valA:.4f}, FDE {valF:.4f})"
            f"{'  [best]' if valL == best_val else ''}"
        )

    net.load_state_dict(torch.load(CKPT_OUT, map_location=device))
    testL, testA, testF = eval_loader(net, test_loader, device, lam=lam)
    print(f"\nTEST | L {testL:.4f} | ADE {testA:.4f} | FDE {testF:.4f}")
    print(f"Saved: {CKPT_OUT}")


if __name__ == "__main__":
    main()
