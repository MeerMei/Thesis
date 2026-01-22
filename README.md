# HighD Trajectory Prediction (LSTM / LSTM+MNL / LSTM+MNL+VAE)

This repository contains three PyTorch scripts to train/evaluate trajectory prediction models on HighD `.mat` splits produced by the MATLAB preprocessor.

## Models

- **Baseline-1 (LSTM-only)**: `baseline_lstm_full.py`  
  LSTM encoder over ego history + MLP trajectory head.

- **Baseline-2 (LSTM + MNL)**: `baseline_lstm_mnl.py`  
  Shared LSTM encoder.  
  - Trajectory head: same as Baseline-1  
  - Decision head: MNL-style linear logits over `[phi(x_t), h_t]`

- **Proposed (LSTM + MNL + VAE)**: `proposed_lstm_mnl_vae.py`  
  Shared LSTM encoder + VAE latent behavior `z_t`.  
  - Decision head uses `[phi(x_t), h_t, z_t]`  
  - Trajectory head uses intent-aware decoding (see code)

## Data format expected

Your `.mat` files should contain:
- `traj`: rows with at least `[datasetId, vehicleId, frameId, ...]`
- `tracks`: MATLAB cell array `{datasetId, vehicleId}` → per-vehicle matrix with:
  - col0: frame id
  - col1: x
  - col2: y
  - last col: maneuver class id (needed for Baseline-2 and Proposed)
- `historical_length` (H), `future_length` (F)

## Install

```bash
pip install -r requirements.txt


