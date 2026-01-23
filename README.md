# HighD Trajectory Prediction (LSTM / LSTM+MNL / LSTM+MNL+VAE)

This repository contains three PyTorch scripts to **train** trajectory prediction models on **HighD** `.mat` splits produced by a MATLAB preprocessor, plus one PyTorch script to **evaluate and compare** all three models.

## Models

### Baseline‑1 (LSTM‑only) — `baseline_lstm_full.py`
- **Encoder:** LSTM over ego history `(H, 2)`.
- **Trajectory head:** MLP predicts `F` future steps `(x, y)`.

### Baseline‑2 (LSTM + MNL) — `baseline_lstm_mnl.py`
- **Shared encoder:** LSTM over ego history.
- **Trajectory head:** same as Baseline‑1.
- **Decision head (MNL‑style):** linear logits over the concatenated vector `[phi(x_t), h_t]`.
  - `phi(x_t)` is a small hand‑crafted, interpretable feature vector derived from history (e.g., last/mean velocity, acceleration, turn angle).

### Proposed (LSTM + MNL + VAE) — `proposed_lstm_mnl_vae.py`
- **Shared encoder:** LSTM over ego history.
- **Latent behaviour:** VAE‑style bottleneck on the encoder state `h_t` to infer a latent embedding `z_t`.
- **Decision head:** linear logits over `[phi(x_t), h_t, z_t]`.
- **Trajectory head:** intent‑aware decoding (see code).

## Data format expected

Your `.mat` split files should contain (at minimum):

- `traj`: a matrix where each row represents a training/evaluation anchor. The first columns are expected to include:
  - `datasetId`, `vehicleId`, `frameId` (MATLAB 1‑based IDs)
- `tracks`: a MATLAB cell array indexed by `{datasetId, vehicleId}` that stores each vehicle’s full time series as a numeric matrix. The scripts assume:
  - column 0: `frameId`
  - column 1: `x`
  - column 2: `y`
  - last column: `maneuver_class_id` (required for Baseline‑2 and Proposed)
- `historical_length` (H), `future_length` (F)
- optional metadata used by the MATLAB preprocessor (e.g., `number_of_agents`, `number_of_features`, etc.)

## Preprocessing (MATLAB)

Use `HighD_preprocess.m` to convert HighD raw CSV tracks into the `.mat` format expected by the Python scripts.

1. Place the HighD raw files (e.g., `XX_tracks.csv` and related meta files) somewhere on disk.
2. Open MATLAB, add this repo to your path, and run the preprocessor.

A typical call looks like:

```matlab
% Example (adjust paths/values to your setup)
historical_length = 30;   % frames
future_length     = 50;   % frames
number_of_agents  = 39;   % max neighbours stored
max_vertical_distance = 50; % meters (used for neighbour filtering)
extra_feature_index = []; % indices of additional track features (optional)

HighD_preprocess(...);
```

The preprocessor should write `TrainSetT.mat`, `ValSetT.mat`, and `TestSetT.mat`.

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Training

The training scripts are **standalone Python files**. Configure them by **editing the path constants at the top** of each script (e.g., `TRAIN_MAT`, `VAL_MAT`, `TEST_MAT`). Then run:

```bash
python baseline_lstm_full.py
python baseline_lstm_mnl.py
python proposed_lstm_mnl_vae.py
```

Each script saves its best checkpoint (by validation loss) in the working directory, typically as:

- `baseline_lstm_best.pt`
- `baseline_lstm_mnl_best.pt`
- `proposed_lstm_mnl_vae_best.pt`

## Evaluation (all 3 models)

Use `eval_three_models.py` to evaluate and compare the three checkpoints on the **test** split.

Example:

```bash
python eval_three_models.py \
  --test_mat ./data/TestSetT.mat \
  --ckpt_lstm ./baseline_lstm_best.pt \
  --ckpt_mnl  ./baseline_lstm_mnl_best.pt \
  --ckpt_vae  ./proposed_lstm_mnl_vae_best.pt \
  --outdir ./eval_out \
  --device cuda \
  --batch 256 \
  --num_workers 2 \
  --bootstrap 1000 \
  --ece_bins 10
```

Key arguments:
- `--test_mat` (required): path to the test `.mat` split.
- `--ckpt_lstm`, `--ckpt_mnl`, `--ckpt_vae` (required): checkpoint paths.
- `--outdir`: where Excel + figures are saved.
- `--device`: `cuda` or `cpu`.
- `--batch`, `--num_workers`, `--pin_memory`: runtime controls.
- `--bootstrap`: number of bootstrap resamples for confidence intervals (0 disables).
- `--ece_bins`: number of bins for reliability diagrams / Expected Calibration Error.




