# MANIFEST — Code_to_submit

Every file and folder copied into this submission package, with a brief
explanation of what it contains and why it is required.

---

## Root-level Python modules

### `model.py`
**What it is:** TensorFlow implementation of the full hybrid architecture.

**Contains:**
- `GroupNorm` — Group Normalisation layer for channels-last tensors `[B, L, C]`.
- `GarchParamNet` — dilated Conv1D stack (5 layers, kernel=3, doubling dilation)
  followed by global average pooling and two FC layers.
  Outputs GARCH parameters: ω (softplus), C = α+β (sigmoid × max_persistence),
  ρ = α/C (sigmoid). Then α = C·ρ, β = C·(1−ρ).
- `GarchVolLayer` — differentiable GARCH(1,1) recursion over the input window;
  produces σ²(t) as a TF computation graph node (gradients flow through it).
- `HybridGarch` — composes `GarchParamNet` + `GarchVolLayer` into one `tf.keras.Model`.

**Why needed:** defines the model architecture used by all TF scripts and End-to-end
notebooks. Must sit at the project root so that `from model import HybridGarch` works
both from root notebooks and from `End-to-end/` notebooks (which add `..` to sys.path).

---

### `engine.py`
**What it is:** TensorFlow training utilities and evaluation functions.

**Contains:**
- `_ReduceLROnPlateau` — minimal learning-rate scheduler (min-mode, factor=0.5).
- `train_epoch` / `validate_epoch` — single-epoch loop over a `tf.data.Dataset`;
  loss = MSE of log-variances; gradient clipping (global norm ≤ 1.0); optional L2
  weight decay applied manually (equivalent to AdamW).
- `train_hybrid_garch` — full training driver: runs N epochs, tracks best val loss,
  saves best weights to a `.weights.h5` checkpoint.
- `eval_log_mse` — compute log-variance MSE on a dataset.
- `arch_baseline_log_mse_per_series` — fits a GARCH(1,1) via `arch` on the training
  portion of each series, then measures log-MSE on the full horizon.
- `qlike` / `log_mse_on_r2` — scalar loss metrics used for real-data evaluation.
- `nn_predict_sigma2_from_window` — predict σ² for a single raw return window
  (handles standardisation / de-standardisation).
- `rolling_forecast_nn` / `rolling_forecast_arch` — rolling one-step-ahead forecasts
  for NN and ARCH(1,1); used in the baseline comparison section of the notebooks.

**Why needed:** imported by `test_on_real_time_series.ipynb`, `Portfolio.ipynb`, and
`End-to-end/generate_last_model_diagnostic.py` for training, evaluation, and baseline
computation.

---

### `data_utils.py`
**What it is:** Synthetic data generation and TF dataset pipeline.

**Contains:**
- `GARCHGenerator` — generates a single GARCH(1,1) return series with configurable
  (ω, α, β) and burn-in; outputs `(returns, variances)` as `tf.Tensor`.
- `compute_garch_variance_from_params` — re-computes σ²[t] from known parameters
  (used for sanity-check / labelling).
- `MultiGARCHDataset` — sliding-window dataset over multiple series;
  `.to_tf_dataset()` returns a shuffled, batched `tf.data.Dataset` of
  `(x=[B,W,1], y=[B])` pairs.
- `sample_params_by_C` — samples GARCH parameters via the persistence
  parameterisation C = α+β, ρ = α/C, v̄ = ω/(1−C) to ensure stationarity.
- `make_synth_dataset` — convenience wrapper: samples params → generates all series
  → stacks into matrices.
- `split_series_indices` — random train/val/test split by series index.
- `standardize_returns` / `scale_variances` — z-score helpers for pre-processing.

**Why needed:** provides the synthetic training data (Step 1) and all dataset
infrastructure for the TF training pipeline.

---

### `data_stocks_WIKI_price.py`
**What it is:** Real market data loader.

**Contains:**
- `get_cleaned_data(file_path, train_ratio, min_price, max_consec_zeros)` — loads the
  WIKI Prices ZIP/CSV, pivots to an `adj_close` matrix, computes log/percentage
  returns, and applies two no-look-ahead filters:
  1. price ≥ `min_price` on the first test day (default $3),
  2. at most `max_consec_zeros` consecutive zero-return days (default 20).
  Returns a `pd.DataFrame` of shape `[T, N_stocks]` with daily returns.

**Why needed:** used by every notebook that touches real WIKI data
(`test_on_real_time_series.ipynb`, `Portfolio.ipynb`, all `End-to-end/` notebooks,
and `generate_last_model_diagnostic.py`).

---

### `visualization.py`
**What it is:** Diagnostic and result plotting utilities (TensorFlow-aware).

**Contains:**
- `plot_predictions_analysis` — 2×2 figure: predicted vs true variance scatter,
  residual histogram, α distribution, β distribution.
- `rolling_params` — batch-computes ω, α, β, C over rolling windows of one series.
- `plot_persistence_over_time` — plots C = α+β through time for one series.
- `collect_true_pred_sigma2` — iterates a `tf.data.Dataset` and collects
  ground-truth and predicted variances into arrays.
- `plot_true_pred_scatter` — scatter plot with optional log-log scale and
  log-MSE / log-correlation statistics.
- `plot_training_curves` — train/val loss over epochs with optional test-loss line.

**Why needed:** imported by `test_on_real_time_series.ipynb` for training diagnostics
and by `Portfolio.ipynb` for evaluation plots.

---

## Notebooks

### `GARCH_NN_Synthetic_Test.ipynb`
**Framework:** PyTorch (self-contained — all classes defined inline).

**What it does:**
- Defines `GARCHGenerator`, `MultiGARCHDataset`, `GarchParamNet`, `HybridGarch`,
  `GarchVolLayer` (PyTorch versions) directly in cells.
- Samples 2 000 random GARCH parameter sets and generates 2 000-step synthetic series.
- Trains for 10 epochs (log-variance MSE loss, AdamW optimiser).
- Evaluates extrapolation to out-of-distribution persistence values.
- Benchmarks against the `arch` GARCH(1,1) baseline.
- Saves `hybrid_garch_pretrained_synth_full.pt`.

**Why needed:** Step 1 of the pipeline — the synthetic pre-training that gives the
model a structural prior before it sees any real data.

---

### `test_on_real_time_series.ipynb`
**Framework:** PyTorch (imports `model.py`, `engine.py`, `data_stocks_WIKI_price.py`,
`visualization.py`).

**What it does:**
- Loads and cleans WIKI Prices.
- Defines a random-batch generator for real data (no full materialization).
- Compares training from scratch vs fine-tuning from `hybrid_garch_pretrained_synth_full.pt`.
- Applies zero-aware fine-tuning (filter: <20 % zeros in window, r² ≥ 1e-8).
- Evaluates against GARCH(1,1), EWMA, naïve, and constant-variance baselines.
- Saves `finetune_real_zeroaware_r2min1e8_best.pt`.

**Why needed:** Step 2 — produces the final fine-tuned model checkpoint used
by all portfolio notebooks.

---

### `Portfolio.ipynb`
**Framework:** PyTorch (imports `model.py`, `engine.py`, `data_stocks_WIKI_price.py`,
`visualization.py`).

**What it does:**
- Loads `finetune_real_zeroaware_r2min1e8_best.pt`.
- Computes daily σ² for 100 WIKI stocks over the test period using four methods:
  CNN-GARCH, historical volatility, EWMA, and GARCH(1,1).
- Builds inverse-variance portfolios for each method and the equal-weight benchmark.
- Reports Sharpe ratio, annualised return / volatility, and maximum drawdown.
- Runs a **Monte Carlo Wilcoxon** test (200 iterations × 30 random stocks) to
  check statistical significance of NN-GARCH's variance advantage over ARCH.
- Saves `mc_results.csv` to cache the Monte Carlo results.

**Why needed:** Step 3 — the main financial evaluation of the model. Contains the
Wilcoxon table and portfolio plots used in the project report.

---

## Checkpoints

### `hybrid_garch_pretrained_synth_full.pt`
**Format:** PyTorch state dict + metadata dict:
```python
{
  "model_state": ...,
  "window_size": 90,
  "hidden_dim":  32,
  "mean_r":      <float>,   # training-set mean of standardised returns
  "std_r":       <float>,   # training-set std
}
```
**Why needed:** starting point for fine-tuning in `test_on_real_time_series.ipynb`.
Without it, fine-tuning degrades to training from scratch, removing the synthetic
prior benefit shown in the comparison cells.

---

### `finetune_real_zeroaware_r2min1e8_best.pt`
**Format:** PyTorch state dict (best validation checkpoint from zero-aware fine-tuning).

**Why needed:** the **final model** used by `Portfolio.ipynb` and referenced as the
baseline in all End-to-end portfolio notebooks. Loading this file is the prerequisite
for every evaluation and portfolio construction step.

---

## Small data files

### `mc_results.csv`
**What it is:** cached output of the 200-iteration Monte Carlo loop in `Portfolio.ipynb §12`.

**Columns:** `iter`, `sharpe_nn`, `sharpe_arch`, `vol_nn`, `vol_arch`, `ret_nn`,
`ret_arch`, `dd_nn`, `dd_arch` (and possibly derived columns).

**Why needed:** the Monte Carlo loop takes ~15–30 minutes. The CSV lets the kernel
be restarted and the Wilcoxon analysis re-run instantly by loading the cached results:
```python
mc_df = pd.read_csv('mc_results.csv')
```

---

## Dependency files

### `requirements.txt`
Lists all packages with exact versions as installed in the development environment
(Python 3.12). Install with `pip install -r requirements.txt`.

---

## End-to-end/

### `End-to-end/generate_last_model_diagnostic.py`
**Framework:** TensorFlow.

**What it does:** loads the TF model from
`finetune_real_zeroaware_r2min1e8_best.weights.h5`, samples 5 000 filtered real
test windows, predicts σ², and produces `nn_garch_real_test_diagnostic.png`
(hexbin scatter + volatility-ratio histogram).

**Why needed:** standalone evaluation script — generates the main model diagnostic
figure for the report without opening a notebook.

> **Note:** requires `finetune_real_zeroaware_r2min1e8_best.weights.h5` (TF format)
> at the project root. See the TF checkpoint note in `README.md`.

---

### `End-to-end/portfolio_optimization.ipynb`
**Framework:** TensorFlow (no PyTorch).

**What it does:** baseline End-to-end GMV comparison with **three strategies**:
1. **GMV-GARCH** — GARCH(1,1) vols + raw rolling correlation → cvxpy (constrained).
2. **GMV-NN** — CNN-GARCH vols + raw rolling correlation → cvxpy (constrained).
3. **GMV-NN-Clean** — CNN-GARCH vols + RIENet-cleaned correlation → cvxpy (constrained).

Uses `rienet` (TF package). Does not use `rienet-torch`.

**Why needed:** primary TF-path GMV notebook; shows the clean three-way comparison
used in the report figures.

---

### `End-to-end/portfolio_optimization_rienet.ipynb`
**Framework:** TensorFlow + PyTorch (mixed).

**What it does:** same three GMV strategies as `portfolio_optimization.ipynb`,
but the RIENet layer uses `rienet_torch` (`CorrelationEigenTransformLayer` +
`EigenWeightsLayer`) for the GMV-NN-Clean strategy.

Constrained weights (long-only, max 10 % per stock, solved via cvxpy).

**Why needed:** demonstrates integration of the PyTorch RIENet package into the
TF-based covariance pipeline; produces constrained GMV results with eigen-cleaning.

---

### `End-to-end/portfolio_unconstrained_rienet.ipynb`
**Framework:** TensorFlow + PyTorch (mixed).

**What it does:** same three strategies as above, but with **unconstrained** GMV
weights solved analytically via Cholesky inversion:
```
w* = Σ⁻¹ · 1 / (1ᵀ · Σ⁻¹ · 1)
```
No cvxpy required for GMV-GARCH and GMV-NN; `EigenWeightsLayer` (rienet_torch)
is used for GMV-NN-Clean.

**Why needed:** provides a fair comparison where the only difference between
strategies is the quality of the covariance estimate, not the solver constraints.
This is the notebook for the unconstrained GMV results in the report.

---

## Files intentionally excluded

| Excluded | Reason |
|---|---|
| `.venv/` | Virtual environment — install from `requirements.txt` instead |
| `__pycache__/` | Python bytecode cache |
| `finetune_real_best.pt` | Intermediate checkpoint (without zero-aware filter) — superseded by final |
| `finetune_real_fixednorm_best.pt` | Intermediate checkpoint (fixed-norm variant) — superseded |
| `finetune_real_zeroaware_best.pt` | Earlier zero-aware checkpoint (no r² floor) — superseded |
| `hybrid_garch_real.pt` | Scratch-trained real-data checkpoint — for comparison only |
| `scratch_real_best.pt` | Scratch-trained baseline — not needed to run the main pipeline |
| `Main.ipynb` | Obsolete early prototype notebook (PyTorch, pre-dates modular scripts) |
| `report/`, `End-to-end/report/` | LaTeX report source and compiled PDF |
| `End-to-end/*.aux`, `*.log`, `*.nav`, `*.toc`, `*.snm` | LaTeX compilation artefacts |
| `End-to-end/*.png` (output figures) | Generated outputs — reproducible by running the notebooks |
| `*.png` at root | Diagnostic outputs — reproducible by running notebooks |
| `WIKI_PRICES_Raccourci.lnk` | Windows shortcut to external data file |
| `Ressource/` | Reference papers (external PDFs) |
| `Nouveau dossier/` | Draft report PDF |
| `.claude/` | Claude Code IDE settings |
