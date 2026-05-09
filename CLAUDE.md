# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ball-conditioned trajectory prediction for sports. A transformer model predicts player trajectories by treating ball position as a privileged cross-attention signal (asymmetric: ball → players) rather than as just another agent. Evaluated against Kalman filter, LSTM, and symmetric transformer baselines. Uses SoccerNet data (originally designed for SportsMOT but pivoted).

## Commands

```bash
# Activate environment
source .venv/bin/activate

# Install (editable)
pip install -r requirements.txt && pip install -e .

# Run tests
pytest tests/ -q

# Train a single model
python -m ballcond.train --config configs/synthetic_ballcond.yaml --out results

# Run all 4 models on synthetic data
bash scripts/run_synthetic_sweep.sh
```

## Architecture

### Data Flow
`Sequence` → `WindowDataset` (sliding windows) → `collate_windows` (batch with padding) → Model → `(B, F, N, 2)` predictions

- Positions are normalized to [0,1]. Scale factors stored for denormalization.
- Windows have variable player counts; `collate_windows` pads and produces `player_pad_mask`.
- All models predict velocity deltas that are cumsum'd to positions.

### Models (`src/ballcond/models/`)
- **kalman.py** — Non-learned constant-velocity Kalman filter baseline (no nn.Module)
- **lstm.py** — Per-player LSTM, no inter-agent or ball context
- **transformer_symmetric.py** — Ball treated as ordinary agent with type embedding; factored time+agent self-attention
- **transformer_ballcond.py** — **Main model.** Ball encoded separately via `_BallEncoder`, then injected into player representations via cross-attention in each `_BallCondLayer`. Falls back gracefully when ball is absent (zero memory).

### Shared Blocks (`models/blocks.py`)
- `SinusoidalPositionalEncoding` — unlearned 1D PE along time
- `TrajectoryTokenizer` — projects (x,y) + velocity delta → d_model

### Data (`src/ballcond/data/`)
- **synthetic.py** — Generates ball-driven player trajectories with configurable attraction/noise. Good for local dev.
- **sportsmot.py** — MOT17-format loader. Note: SportsMOT lacks ball annotations; needs external ball source or alternative dataset (SoccerNet-Tracking, NBA SportVU).
- **windows.py** — `WindowDataset` extracts (history, future) sliding windows; filters by `min_players`.
- **types.py** — `Sequence` and `Window` dataclasses.

### Training (`src/ballcond/train.py`)
Entry point. Loads config via OmegaConf, builds dataset/model, runs train loop with `masked_mse` loss, evaluates ADE/FDE at horizons [5, 10, 20]. Saves `metrics.json`, `model.pt`, `config.yaml` to output dir.

### Metrics (`src/ballcond/metrics.py`)
- ADE (average displacement error) and FDE (final displacement error) at configurable horizons
- All metrics are masked to only count valid player-frame pairs

## Config System
YAML configs in `configs/` parsed with OmegaConf. Key fields: `data.name` (synthetic|sportsmot), `model.name` (kalman|lstm|transformer_symmetric|transformer_ballcond), `model.kwargs` (d_model, n_heads, n_layers, dropout), `data.history`/`data.horizon` (default 20/20).

## Formatting
Uses `black` with line length 100.
