# # HighD Trajectory Prediction Models (LSTM / LSTM+MNL / LSTM+MNL+VAE)

This repository contains three **minimal, reproducible** PyTorch models for **highway trajectory prediction** on the HighD dataset, using the `.mat` splits produced by your MATLAB preprocessing script.

## Models

### Baseline-1: LSTM-only (trajectory)
**File:** `baseline_lstm_full.py`

- Encoder: LSTM over ego history `(H,2)`
- Decoder: MLP predicts future positions `(F,2)`
- Loss: `ADE + λ·FDE`

### Baseline-2: LSTM + MNL (trajectory + maneuver)
**File:** `baseline_lstm_mnl.py`

- Shared encoder: LSTM → hidden state `h_t`
- Trajectory head: same as Baseline-1
- Decision head (MNL-style): linear logits from `[φ(x_t), h_t]`
- Multi-task loss: `L = L_traj + α·CE`
- Interpretability: exports decision-layer coefficients (β, ASC)

### Proposed: LSTM + MNL + VAE (latent behavior)
**File:** `proposed_lstm_mnl_vae.py`

- Shared encoder: LSTM → `h_t`
- VAE bottleneck: `(μ, logσ)=MLP(h_t)`, `z=μ+σ⊙ε`
- Decision head: logits from `[φ(x_t), h_t, z_t]`
- Trajectory head:
  - **Option A (default):** single decoder `f(h_t, z_t)`
  - **Option B (optional):** mixture-of-experts (one expert per maneuver), weighted by class probabilities
- Loss: `L = L_traj + α·CE + γ·KL` with **KL annealing**
- ✅ Includes **robust checkpoint loading** that can infer `z_dim` / decoder mode from a checkpoint to avoid shape mismatch errors.

---

## Data format

All scripts expect HighD `.mat` files produced by your MATLAB preprocessor (not included here).

Required keys inside each `.mat` file:

- `traj`: `(N, K)` matrix, one row per anchor sample. First columns contain:
  `datasetId, vehicleId, frameId, ...`
- `tracks`: 2D cell-like array indexed by `{datasetId, vehicleId}` (MATLAB 1-based),
  where each cell stores a `(T, cols)` track matrix.
  The first columns must be:
  - `frameId` (col 0)
  - `x` (col 1)
  - `y` (col 2)
  - **maneuver class id as the last column** (required for Baseline-2 and Proposed)
- `historical_length` (H), `future_length` (F)
- `number_of_maneuvers` (optional, default `9`)

> **Note:** If your `.mat` file is saved as MATLAB **v7.3**, `scipy.io.loadmat` cannot read it.  
> Re-save as **v7** or implement an `h5py` loader.

---

## Setup

Create an environment and install dependencies:

```bash
pip install numpy scipy torch
```

Optional (only if you use plotting/evaluation scripts in your project):

```bash
pip install matplotlib pandas openpyxl
```

---

## Training & evaluation

### Baseline-1

```bash
python baseline_lstm_full.py \
  --train_mat ./data/TrainSetT.mat \
  --val_mat   ./data/ValSetT.mat \
  --test_mat  ./data/TestSetT.mat \
  --ckpt ./baseline_lstm_best.pt
```

Evaluate only:

```bash
python baseline_lstm_full.py \
  --train_mat ./data/TrainSetT.mat --val_mat ./data/ValSetT.mat --test_mat ./data/TestSetT.mat \
  --ckpt ./baseline_lstm_best.pt --eval_only
```

### Baseline-2 (LSTM+MNL)

```bash
python baseline_lstm_mnl.py \
  --train_mat ./data/TrainSetT.mat \
  --val_mat   ./data/ValSetT.mat \
  --test_mat  ./data/TestSetT.mat \
  --ckpt ./baseline_lstm_mnl_best.pt \
  --coeff_csv ./mnl_coefficients.csv
```

### Proposed (LSTM+MNL+VAE)

Option A (single decoder, default):

```bash
python proposed_lstm_mnl_vae.py \
  --train_mat ./data/TrainSetT.mat \
  --val_mat   ./data/ValSetT.mat \
  --test_mat  ./data/TestSetT.mat \
  --ckpt ./proposed_lstm_mnl_vae_best.pt \
  --z_dim 16 \
  --traj_mode single
```

Option B (mixture-of-experts):

```bash
python proposed_lstm_mnl_vae.py \
  --train_mat ./data/TrainSetT.mat \
  --val_mat   ./data/ValSetT.mat \
  --test_mat  ./data/TestSetT.mat \
  --ckpt ./proposed_lstm_mnl_vae_best.pt \
  --z_dim 16 \
  --traj_mode moe
```

Evaluate only:

```bash
python proposed_lstm_mnl_vae.py \
  --train_mat ./data/TrainSetT.mat --val_mat ./data/ValSetT.mat --test_mat ./data/TestSetT.mat \
  --ckpt ./proposed_lstm_mnl_vae_best.pt \
  --eval_only
```

