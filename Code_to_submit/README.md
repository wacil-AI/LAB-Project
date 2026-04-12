# CNN-GARCH + RIENet — Volatility Forecasting & GMV Portfolio Optimisation

## Project overview

This project develops a **hybrid CNN-GARCH neural network** for financial volatility
forecasting, and uses its predictions to construct a **Global Minimum Variance (GMV)
portfolio** that out-performs classical baselines.

### Core idea

A dilated Conv1D network reads a window of past returns and outputs three GARCH(1,1)
parameters (ω, α, β). Those parameters are passed to a differentiable GARCH recursion
layer that produces a one-step-ahead variance forecast.  
This hybrid design keeps the GARCH structural prior while letting the network adapt
the parameters to each regime from data alone.

```
returns[t-W : t]  →  [Conv1D × 5 + GroupNorm + LeakyReLU]  →  GAP
                  →  FC → (ω, C=α+β, ρ)
                  →  GARCH recursion  →  σ²(t)
```

Training is done on **synthetic** GARCH(1,1) series (no look-ahead bias), then
**fine-tuned** on real WIKI stock-price returns using a zero-aware data filter.

The predicted volatilities feed a GMV portfolio via a rolling covariance matrix.
A **RIENet** (Riemannian Information-Efficient Estimator) layer optionally cleans
the rolling sample correlation matrix by eigenvalue shrinkage before the optimisation.

---

## Folder structure

```
Code_to_submit/
│
│  ── Python library modules ──────────────────────────────────────────────────
├── model.py                                  # Architecture: HybridGarch (TensorFlow)
├── engine.py                                 # Training loop, evaluation, baselines
├── data_utils.py                             # Synthetic GARCH generator & tf.data pipeline
├── data_stocks_WIKI_price.py                 # Real WIKI data loader / cleaner
├── visualization.py                          # Diagnostic & result plots
│
│  ── Notebooks (run in order) ────────────────────────────────────────────────
├── GARCH_NN_Synthetic_Test.ipynb             # Step 1 – train on synthetic data (PyTorch)
├── test_on_real_time_series.ipynb            # Step 2 – fine-tune on real returns (PyTorch)
├── Portfolio.ipynb                           # Step 3 – portfolio eval + Monte Carlo (PyTorch)
│
│  ── Saved checkpoints ───────────────────────────────────────────────────────
├── hybrid_garch_pretrained_synth_full.pt     # output of Step 1 (synthetic pre-train)
├── finetune_real_zeroaware_r2min1e8_best.pt  # output of Step 2 (FINAL model)
│
│  ── Small data file ─────────────────────────────────────────────────────────
├── mc_results.csv                            # Monte Carlo Wilcoxon results (Step 3)
│
│  ── Dependencies ────────────────────────────────────────────────────────────
├── requirements.txt
├── README.md                                 # this file
├── MANIFEST.md                               # per-file description
│
│  ── End-to-end TF portfolio notebooks ───────────────────────────────────────
└── End-to-end/
    ├── generate_last_model_diagnostic.py     # diagnostic plot script (TF)
    ├── portfolio_optimization.ipynb          # GMV constrained – no RIENet (TF)
    ├── portfolio_optimization_rienet.ipynb   # GMV constrained + RIENet  (TF + PyTorch)
    └── portfolio_unconstrained_rienet.ipynb  # GMV unconstrained + RIENet (TF + PyTorch)
```

> **Two ML frameworks are used.**  
> The three root-level notebooks (`GARCH_NN_Synthetic_Test`, `test_on_real_time_series`,
> `Portfolio`) use **PyTorch** and load `.pt` checkpoints.  
> The TF library modules (`model.py`, `engine.py`, `data_utils.py`) and the four
> `End-to-end/` files use **TensorFlow** and expect a `.weights.h5` checkpoint.  
> See [TF checkpoint note](#tf-checkpoint-note) below.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.12 was used during development.

Key packages:

| Package | Version | Role |
|---|---|---|
| `tensorflow` | 2.20.0 | TF model / End-to-end notebooks |
| `torch` | 2.10.0 | Main training notebooks |
| `numpy` | 1.26.4 | numerical arrays |
| `pandas` | 2.3.3 | data loading |
| `matplotlib` | 3.10.8 | plots |
| `scipy` | 1.15.3 | statistics |
| `arch` | 8.0.0 | GARCH(1,1) baseline |
| `cvxpy` | 1.8.2 | constrained GMV optimisation |
| `rienet` | 1.1.3 | RIENet TF layer |
| `rienet-torch` | 0.1.2 | RIENet PyTorch layer |

The `End-to-end/` notebooks also run `pip install rienet_torch` / `pip install rienet`
in their first cell; this is safe to re-run.

---

## External data

All notebooks require the **WIKI Prices** dataset (Quandl / Nasdaq Data Link):

```
WIKI_PRICES_212b326a081eacca455e13140d7bb9db.zip
```

Set the variable `FILE_PATH` (or `DATA_PATH`) near the top of each notebook to the
local path of this file.

The loader `data_stocks_WIKI_price.get_cleaned_data()` applies two filters with no
look-ahead bias:

1. Price ≥ $3 on the first test day.
2. At most 20 consecutive zero-return days (proxy for delisted / illiquid periods).

After filtering, the universe is typically ~1 390 stocks over ~4 500 trading days
(2000-01-01 onwards).

---

## Full pipeline — step by step

### Step 1 — Train on synthetic GARCH(1,1) data

**Notebook:** `GARCH_NN_Synthetic_Test.ipynb`

This notebook is **self-contained** (model, dataset, training loop all defined
inline in PyTorch). It does not import `model.py` / `engine.py`.

What it does:

1. Samples 2 000 sets of GARCH parameters (ω, α, β) uniformly:
   - persistence C = α + β ∈ [0.05, 0.97]
   - split ratio ρ = α / C ∈ [0.05, 0.95]
   - unconditional variance v̄ ∈ [0.5, 5.0]
2. Generates 2 000-step synthetic return series for each (with 200-step burn-in).
3. Builds sliding-window datasets (W = 90, stride = 10).
4. Trains the hybrid CNN-GARCH model for 10 epochs; loss = log-variance MSE.
5. Saves the best checkpoint to `hybrid_garch_pretrained_synth_full.pt`.

Key hyper-parameters in the notebook:

```python
N_SERIES   = 2000        # number of synthetic time series
N_SAMPLES  = 2000        # length of each series
WINDOW_SIZE = 90         # CNN input width
HIDDEN_DIM  = 32         # Conv1D / FC width
N_EPOCHS    = 10
LR          = 1e-4
```

**TF equivalent:** `data_utils.py` + `model.py` + `engine.py` expose identical
functionality (`GARCHGenerator`, `MultiGARCHDataset`, `train_hybrid_garch`).
Running `engine.train_hybrid_garch(model, train_loader, val_loader, ...)` in a
TF script reproduces the same pipeline.

---

### Step 2 — Fine-tune on real WIKI stock returns

**Notebook:** `test_on_real_time_series.ipynb`

Imports `model.py`, `engine.py`, `data_stocks_WIKI_price.py`, and `visualization.py`.

What it does:

1. Loads and cleans WIKI Prices (via `get_cleaned_data`).
2. Splits into train (rows 0–1 200) / val (1 200–1 400) / test (1 400–end).
3. Loads the synthetic pre-train checkpoint `hybrid_garch_pretrained_synth_full.pt`.
4. Fine-tunes with a **zero-aware filter**: each training sample must satisfy
   - fewer than 20 % zero returns in its 90-day window, **and**
   - realized squared return r² ≥ 1e-8 (avoids fitting to near-zero noise).
5. Uses `train_hybrid_garch` with weight decay (WD = 1e-6) and ReduceLR on plateau.
6. Saves `finetune_real_zeroaware_r2min1e8_best.pt`.

The notebook also benchmarks the fine-tuned model against:
- training from scratch on real data,
- GARCH(1,1) baseline (fitted via `arch`),
- EWMA (λ = 0.94),
- naïve (yesterday's r²),
- constant-variance.

Metrics used: **log-variance MSE**, **QLIKE** (`r²/σ̂² + log σ̂²`),
Pearson / Spearman correlation, and a Wilcoxon paired test.

---

### Step 3 — Portfolio evaluation (PyTorch path)

**Notebook:** `Portfolio.ipynb`

Loads `finetune_real_zeroaware_r2min1e8_best.pt` and builds portfolios on
100 selected WIKI stocks over the out-of-sample test period.

Strategies compared:

| Strategy | Volatility estimate | Weights |
|---|---|---|
| **NN-GARCH** | CNN-GARCH predicted σ² | inverse-variance |
| **Hist-Vol** | 90-day rolling std | inverse-variance |
| **EWMA** | λ=0.94 exponential | inverse-variance |
| **ARCH(1,1)** | fitted GARCH(1,1) | inverse-variance |
| **Equal-Weight** | — | 1/N |

Financial metrics reported:
- Annualised return & volatility
- Sharpe ratio
- Maximum drawdown
- Calmar ratio

**Section 12 — Monte Carlo Wilcoxon test** (`mc_results.csv`):
Repeats the NN vs ARCH experiment 200 times, each time drawing 30 stocks at random
from the full universe.  A Wilcoxon signed-rank test checks whether NN-GARCH
achieves lower portfolio variance than ARCH(1,1) across random sub-universes.
Results are cached in `mc_results.csv` so the kernel can be restarted without
re-running the loop.

---

### Step 3 (TF path) — GMV portfolio with End-to-end notebooks

The `End-to-end/` notebooks use the TF model and build a full
**Global Minimum Variance** portfolio instead of inverse-variance weighting.

Run them in this recommended order:

| Notebook | Description |
|---|---|
| `portfolio_optimization.ipynb` | Constrained GMV (no RIENet) — baseline comparison |
| `portfolio_optimization_rienet.ipynb` | Constrained GMV + RIENet covariance cleaning |
| `portfolio_unconstrained_rienet.ipynb` | Unconstrained (analytical) GMV + RIENet |

All three notebooks:

1. Load the cleaned WIKI data.
2. Select the first `N_STOCKS` (default 50) tickers.
3. Load the TF model from `../finetune_real_zeroaware_r2min1e8_best.weights.h5`.
4. Pre-compute per-stock CNN-GARCH σ² and GARCH(1,1) σ² for every test day.
5. Build rolling covariance matrices and run the GMV solver.
6. Print and plot financial metrics.

> **TF checkpoint note** <a name="tf-checkpoint-note"></a>  
> The End-to-end notebooks require `finetune_real_zeroaware_r2min1e8_best.weights.h5`
> (TensorFlow `.h5` format) at the **project root** (one level above `End-to-end/`).
> This file was generated during development but is not included here.  
> To regenerate it, run the TF training pipeline using `model.py`, `engine.py`,
> and `data_utils.py`, then fine-tune with `engine.train_hybrid_garch(...)` and
> save with `model.save_weights("finetune_real_zeroaware_r2min1e8_best.weights.h5")`.  
> The equivalent **PyTorch** checkpoint `finetune_real_zeroaware_r2min1e8_best.pt`
> is included and works directly with `Portfolio.ipynb`.

---

### Step 4 — Model diagnostic on real test data

```bash
cd Code_to_submit
python End-to-end/generate_last_model_diagnostic.py
```

Requires the `.weights.h5` TF checkpoint and the WIKI data ZIP.
Produces `End-to-end/nn_garch_real_test_diagnostic.png`:
a hexbin scatter (predicted σ² vs realised r²) and a volatility-ratio histogram.

---

## How to add / use the RIENet layer

RIENet is an **installed package** — no custom source file is needed.

```bash
# TensorFlow version (used in portfolio_optimization.ipynb)
pip install rienet

# PyTorch version (used in portfolio_optimization_rienet.ipynb and portfolio_unconstrained_rienet.ipynb)
pip install rienet-torch
```

### What RIENet does

The raw 90-day sample correlation matrix C̃(t) is noisy when the number of
observations (W = 90) is close to the number of stocks (N ≈ 50).
RIENet replaces the sample eigenvalues with shrinkage estimates derived from
Random Matrix Theory, producing a cleaner estimate Ĉ(t).

### How it is used in the notebooks

```python
from rienet_torch import CorrelationEigenTransformLayer, EigenWeightsLayer

# Instantiate once (q = W/N = window / n_stocks)
corr_layer = CorrelationEigenTransformLayer(n=N_STOCKS, q=WINDOW_SIZE / N_STOCKS)
w_layer    = EigenWeightsLayer(n=N_STOCKS)

# Inside the daily loop:
corr_raw  = rolling_correlation(returns_arr, t, WINDOW_SIZE)   # [N, N]
corr_clean = corr_layer(torch.tensor(corr_raw).unsqueeze(0))   # [1, N, N]
weights    = w_layer(corr_clean, sigma2_diag)                  # [N]  GMV weights
```

`CorrelationEigenTransformLayer` shrinks the eigenvalues of the correlation matrix.  
`EigenWeightsLayer` combines the cleaned correlation with the diagonal volatility
matrix and solves the unconstrained GMV problem analytically.

This constitutes the **GMV-NN-Clean** strategy compared against GMV-GARCH and GMV-NN.

---

## GMV portfolio construction — details

At each test day `t`:

```
1.  σ²_i(t) = CNN-GARCH(returns[t-W:t, i])   for all stocks i
              (or GARCH(1,1) fitted on rolling window for the baseline)

2.  C̃(t)  = sample_correlation(returns[t-W:t, :])   [N×N]
    Ĉ(t)  = RIENet(C̃(t))                            [N×N]  (optional)

3.  D(t)  = diag(σ_i(t))                            [N×N]
    Σ(t)  = D(t) · Ĉ(t) · D(t)                     [N×N]

4a. Constrained GMV (cvxpy):
      w*(t) = argmin  wᵀ Σ(t) w
              s.t.    Σ wᵢ = 1,  wᵢ ≥ 0,  wᵢ ≤ 10%

4b. Unconstrained GMV (analytical):
      w*(t) = Σ(t)⁻¹ · 1  /  (1ᵀ · Σ(t)⁻¹ · 1)

5.  r_portfolio(t) = w*(t)ᵀ · r(t)
```

---

## Reproducing the main results

| Result | Where |
|---|---|
| Training curves (log-variance MSE) | `GARCH_NN_Synthetic_Test.ipynb` §Training |
| Synthetic test MSE vs ARCH baseline | `GARCH_NN_Synthetic_Test.ipynb` §Baseline |
| Fine-tune: scratch vs pre-train | `test_on_real_time_series.ipynb` §Fine-tuning |
| Real-data diagnostic plot | `python End-to-end/generate_last_model_diagnostic.py` |
| Sharpe / vol / drawdown table | `Portfolio.ipynb` §8 |
| Monte Carlo Wilcoxon p-value | `Portfolio.ipynb` §12 (or load `mc_results.csv`) |
| GMV constrained comparison | `End-to-end/portfolio_optimization_rienet.ipynb` §10 |
| GMV unconstrained comparison | `End-to-end/portfolio_unconstrained_rienet.ipynb` §10 |

All seeds are set at the top of each notebook (`SEED = 42`). Results are
fully reproducible given the same WIKI Prices data file.
