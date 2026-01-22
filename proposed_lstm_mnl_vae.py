# proposed_lstm_mnl_vae.py
# Proposed: LSTM + MNL decision + VAE-style latent behavior z + intent-aware trajectory head.

import os
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
CKPT_OUT  = "./proposed_lstm_mnl_vae_best.pt"


# -------------------------
# Hyperparameters (match your trained checkpoint!)
# -------------------------
HID_DIM   = 128
PHI_DIM   = 8
Z_DIM     = 16          # must match checkpoint (your error indicates 16)
TRAJ_MODE = "moe"       # must match checkpoint if you trained with MoE


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset (same as Baseline-2)
# -------------------------
class EgoChoiceDataset(Dataset):
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
            theta = float(np.arctan2(sinang, cosang))

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
# Model
# -------------------------
class ProposedLSTM_MNL_VAE(nn.Module):
    """
    LSTM encoder -> latent z (mu, log_sigma) -> reparameterize
    Decision head (MNL-style): beta(phi) + u(h) + v(z) + ASC
    Trajectory head:
      - single: f([h,z]) -> (F,2)
      - moe   : sum_i P(i)*expert_i([h,z])
    """

    def __init__(self, hist: int, fut: int, num_classes: int, in_dim: int = 2,
                 hid: int = HID_DIM, phi_dim: int = PHI_DIM, z_dim: int = Z_DIM, traj_mode: str = TRAJ_MODE):
        super().__init__()
        self.H = hist
        self.F = fut
        self.C = num_classes
        self.hid = hid
        self.z_dim = z_dim
        self.traj_mode = traj_mode

        self.enc = nn.LSTM(input_size=in_dim, hidden_size=hid, batch_first=True)

        self.latent_stem = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(inplace=True))
        self.mu_head     = nn.Linear(hid, z_dim)
        self.logsig_head = nn.Linear(hid, z_dim)

        self.beta_phi = nn.Linear(phi_dim, num_classes, bias=False)
        self.u_h      = nn.Linear(hid,      num_classes, bias=False)
        self.v_z      = nn.Linear(z_dim,    num_classes, bias=False)
        self.asc      = nn.Parameter(torch.zeros(num_classes))

        if traj_mode == "single":
            self.traj_head = nn.Sequential(
                nn.Linear(hid + z_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, fut * 2),
            )
        elif traj_mode == "moe":
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hid + z_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, fut * 2),
                )
                for _ in range(num_classes)
            ])
        else:
            raise ValueError("traj_mode must be 'single' or 'moe'")

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        # mean KL per batch
        kl = 0.5 * (torch.exp(log_sigma) + mu**2 - 1.0 - log_sigma).sum(dim=-1)
        return kl.mean()

    def forward(self, ego_hist: torch.Tensor, phi: torch.Tensor):
        B = ego_hist.size(0)
        _, (h_n, _) = self.enc(ego_hist)
        h = h_n.squeeze(0)  # (B,hid)

        stem = self.latent_stem(h)
        mu = self.mu_head(stem)
        log_sig = self.logsig_head(stem)
        z = self.reparameterize(mu, log_sig)

        logits = self.beta_phi(phi) + self.u_h(h) + self.v_z(z) + self.asc
        probs = torch.softmax(logits, dim=-1)

        inp = torch.cat([h, z], dim=-1)

        if self.traj_mode == "single":
            traj = self.traj_head(inp).view(B, self.F, 2)
        else:
            outs = [exp(inp).view(B, self.F, 2) for exp in self.experts]   # list of (B,F,2)
            stacked = torch.stack(outs, dim=1)                             # (B,C,F,2)
            w = probs.view(B, self.C, 1, 1)
            traj = (w * stacked).sum(dim=1)                                # (B,F,2)

        return traj, logits, probs, mu, log_sig


# -------------------------
# Loss
# -------------------------
def ade_fde(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    diff = pred - gt
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


def ade_fde_loss(pred: torch.Tensor, gt: torch.Tensor, lam: float = 1.0):
    ade, fde = ade_fde(pred, gt)
    return ade + lam * fde, ade, fde


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device, lam: float, alpha: float, gamma: float):
    net.eval()
    Ls, Ltrs, As, Fs, CEs, KLs, Accs = [], [], [], [], [], [], []
    for b in loader:
        hist = b["ego_hist"].to(device)
        fut  = b["ego_fut"].to(device)
        phi  = b["phi"].to(device)
        y    = b["choice"].to(device)

        traj, logits, _, mu, log_sig = net(hist, phi)

        Ltr, ADE, FDE = ade_fde_loss(traj, fut, lam=lam)
        CE = F.cross_entropy(logits, y, reduction="mean")
        KL = net.kl_divergence(mu, log_sig)
        L  = Ltr + alpha * CE + gamma * KL

        acc = (logits.argmax(-1) == y).float().mean()

        Ls.append(L.item()); Ltrs.append(Ltr.item())
        As.append(ADE.item()); Fs.append(FDE.item())
        CEs.append(CE.item()); KLs.append(KL.item())
        Accs.append(acc.item())

    return {
        "loss": float(np.mean(Ls)),
        "traj_loss": float(np.mean(Ltrs)),
        "ADE": float(np.mean(As)),
        "FDE": float(np.mean(Fs)),
        "CE": float(np.mean(CEs)),
        "KL": float(np.mean(KLs)),
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

    net = ProposedLSTM_MNL_VAE(
        hist=train_ds.H, fut=train_ds.F, num_classes=train_ds.num_m,
        in_dim=2, hid=HID_DIM, phi_dim=PHI_DIM, z_dim=Z_DIM, traj_mode=TRAJ_MODE
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    lam = 1.0
    alpha = 1.0
    gamma_max = 0.1
    warmup_epochs = 10

    best_val = float("inf")
    epochs = 20

    for ep in range(1, epochs + 1):
        net.train()
        gamma = gamma_max * min(1.0, ep / max(1, warmup_epochs))

        trL, trA, trF, trCE, trKL, trAcc = [], [], [], [], [], []

        for b in train_loader:
            hist = b["ego_hist"].to(device)
            fut  = b["ego_fut"].to(device)
            phi  = b["phi"].to(device)
            y    = b["choice"].to(device)

            traj, logits, _, mu, log_sig = net(hist, phi)

            Ltr, ADE, FDE = ade_fde_loss(traj, fut, lam=lam)
            CE = F.cross_entropy(logits, y, reduction="mean")
            KL = net.kl_divergence(mu, log_sig)
            loss = Ltr + alpha * CE + gamma * KL

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            acc = (logits.argmax(-1) == y).float().mean()

            trL.append(loss.item()); trA.append(ADE.item()); trF.append(FDE.item())
            trCE.append(CE.item()); trKL.append(KL.item()); trAcc.append(acc.item())

        val = evaluate(net, val_loader, device, lam=lam, alpha=alpha, gamma=gamma)

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(net.state_dict(), CKPT_OUT)

        print(
            f"Epoch {ep:02d} | "
            f"train L {np.mean(trL):.4f} (ADE {np.mean(trA):.4f}, FDE {np.mean(trF):.4f}, CE {np.mean(trCE):.4f}, KL {np.mean(trKL):.4f}, acc {np.mean(trAcc):.3f}) | "
            f"val L {val['loss']:.4f} (ADE {val['ADE']:.4f}, FDE {val['FDE']:.4f}, CE {val['CE']:.4f}, KL {val['KL']:.4f}, acc {val['acc']:.3f}) | "
            f"gamma={gamma:.3f}"
            f"{'  [best]' if val['loss'] == best_val else ''}"
        )

    net.load_state_dict(torch.load(CKPT_OUT, map_location=device))
    test = evaluate(net, test_loader, device, lam=lam, alpha=alpha, gamma=gamma_max)
    print(f"\nTEST | L {test['loss']:.4f} | ADE {test['ADE']:.4f} | FDE {test['FDE']:.4f} | CE {test['CE']:.4f} | KL {test['KL']:.4f} | acc {test['acc']:.3f}")
    print(f"Saved: {CKPT_OUT}")


if __name__ == "__main__":
    main()
