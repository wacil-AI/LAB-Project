# Lab Project Report Outline

## Target format

- Recommended format: standard report in `article` style, 10 to 15 pages.
- Main language: English.
- Main story line: start from the GMV covariance-estimation problem, justify GARCH, then show why the hybrid CNN-GARCH and RIENet pipeline improves the portfolio construction stage.

## Suggested page budget

1. Title page and abstract: 1 page
2. Introduction: 1.5 to 2 pages
3. State of the art: 1.5 to 2 pages
4. Data and problem formulation: 1 to 1.5 pages
5. Methodology: 2 to 3 pages
6. Numerical results: 2 to 3 pages
7. Ethical and societal impact: 0.5 page
8. Project organization and timeline: 1 page
9. Conclusion and perspectives: 0.5 to 1 page
10. References: 0.5 to 1 page

## Section-by-section writing plan

### 1. Title page

- Project title.
- Student names.
- Supervisor names: include `Christian Bongiorno`.
- Email contacts.
- Date.
- Git repository link with maintainer access.

### 2. Abstract

- Problem: GMV portfolios are highly sensitive to covariance estimation noise.
- Approach: hybrid CNN-GARCH for marginal volatility plus RIENet-based correlation cleaning.
- Data: cleaned WIKI adjusted prices, train/test split 2000-2018.
- Main result: in the 50-stock unconstrained GMV experiment, `GMV-NN-Clean` gives the lowest volatility and `GMV-NN` gives the best Sharpe ratio.
- Robustness: 200 Monte Carlo resamples of 30 stocks; Wilcoxon tests significant for volatility and Sharpe improvements of NN vs GARCH.

### 3. Introduction

- Explain why portfolio variance minimization matters.
- Explain why the covariance matrix is the real bottleneck for GMV.
- Explain why GARCH is a strong baseline:
  daily financial returns exhibit volatility clustering;
  GARCH is parsimonious, interpretable, and financially grounded;
  a one-step variance recursion fits the rolling portfolio setting.
- Explain why pure GARCH can still be limited:
  fixed linear structure, same functional form for all assets, weak ability to capture richer lag interactions.
- Explain the idea of the hybrid model:
  CNN learns nonlinear features from a 90-day return window;
  the GARCH layer keeps the output interpretable and stable.
- Explain why RIENet is relevant:
  even good marginal volatilities are not enough if rolling correlations are noisy;
  covariance cleaning is needed before the GMV inversion step.
- End the introduction with a compact contributions list.

### 4. State of the art

- Modern portfolio theory and GMV: Markowitz mean-variance formulation.
- Volatility forecasting: ARCH and GARCH, then why GARCH(1,1) remains the canonical baseline.
- Covariance estimation and cleaning: sample covariance instability in high dimensions; shrinkage and spectral cleaning motivation.
- Neural portfolio optimization: Bongiorno end-to-end large portfolio optimization through covariance cleaning.
- Parameter-efficient neural GMV under leverage: Bongiorno ICAIF paper.
- Optional addition: add one or two NN volatility forecasting papers if your supervisor expects a broader review.

### 5. Data and problem formulation

- Describe the raw dataset: WIKI adjusted prices.
- Give the final date range used in the cleaned universe.
- Explain the cleaning rules from `data_stocks_WIKI_price.py`:
  remove zero-volume rows;
  adjusted-close returns;
  keep complete series;
  price filter on the first test day;
  remove assets with more than 20 consecutive zero returns.
- Mention the final universe size: `1338` stocks after filters.
- State the 70/30 time split:
  train `2000-01-04` to `2012-10-03`,
  test `2012-10-04` to `2018-03-27`.
- Define notation:
  returns `r_{i,t}`,
  rolling window size `W=90`,
  volatility vector,
  correlation matrix,
  covariance matrix `Sigma_t = D_t C_t D_t`.
- Write the unconstrained GMV formula and, if relevant, the constrained variant.

### 6. Methodology

#### 6.1 Synthetic pretraining

- Explain that the model is first pretrained on simulated GARCH(1,1) series.
- Mention the exploratory synthetic setup from `Main.ipynb`:
  `N_SERIES = 2000`, `N_SAMPLES = 2000`, `WINDOW_SIZE = 90`, train/val/test split `60/20/20`.
- Explain why pretraining helps:
  initialize the model on generic heteroskedastic dynamics before adapting to equities.

#### 6.2 Real-data fine-tuning

- Explain the fine-tuning logic on cleaned WIKI returns.
- Mention the final training choices recovered from `test_on_real_time_series.ipynb`:
  `WINDOW_SIZE = 90`,
  `N_EPOCHS = 20`,
  learning rate around `1e-4`,
  weight decay around `1e-6`,
  zero-aware filtering with `ZERO_RATIO_MAX = 0.20`,
  `r^2` floor around `1e-8`,
  validation checkpoint selection.
- Explain the loss: log-variance MSE.

#### 6.3 Hybrid CNN-GARCH model

- Input: one 90-day return window per asset.
- Architecture:
  5 dilated Conv1D layers,
  kernel size 3,
  dilations `1, 2, 4, 8, 16`,
  GroupNorm and LeakyReLU,
  dense head.
- Mention parameter count used in the final GMV notebooks: `14,499` parameters.
- Explain why 5 layers:
  with kernel 3 and dilations doubling each layer, the receptive field becomes 63 days, while the pooled representation still summarizes the full 90-day window.
- Explain stable parameterization:
  network predicts `omega`, `C`, `rho`,
  then `alpha = C rho`, `beta = C (1-rho)`,
  ensuring `alpha + beta < 1`.
- Write the GARCH recursion.

#### 6.4 RIENet correlation cleaning

- Explain that rolling sample correlations are noisy.
- Explain the spectral idea:
  eigen decomposition,
  clean the eigenvalue structure in a dimension-aware way,
  reconstruct a better conditioned correlation matrix.
- Explain how this enters the GMV pipeline:
  `GMV-GARCH`, `GMV-NN`, `GMV-NN-Clean`.

#### 6.5 Pseudo-code

- Add one compact algorithm for training.
- Add one compact algorithm for daily portfolio construction.

### 7. Numerical results

#### 7.1 Main result: 50-stock unconstrained GMV

- Source notebook: `End-to-end/portfolio_unconstrained_rienet.ipynb`.
- Keep this table in the report:

| Method | Ann. Return (%) | Ann. Vol (%) | Sharpe |
|---|---:|---:|---:|
| GMV-GARCH | 17.39 | 13.95 | 1.2468 |
| GMV-NN | 20.43 | 14.40 | 1.4190 |
| GMV-NN-Clean | 17.53 | 12.75 | 1.3755 |

- Key interpretation:
  `GMV-NN-Clean` minimizes volatility;
  `GMV-NN` maximizes Sharpe.

#### 7.2 Supporting result: constrained GMV with RIENet

- Source notebook: `End-to-end/portfolio_optimization_rienet.ipynb`.
- Keep this table:

| Method | Ann. Return (%) | Ann. Vol (%) | Sharpe |
|---|---:|---:|---:|
| GMV-GARCH | 16.73 | 11.45 | 1.4613 |
| GMV-NN | 19.19 | 11.44 | 1.6781 |
| GMV-NN-Clean | 19.22 | 13.02 | 1.4756 |

- Interpretation:
  in this constrained 30-stock experiment, `GMV-NN` dominates on Sharpe and volatility.

#### 7.3 Supporting result: 100-stock constrained comparison

- Source notebook: `End-to-end/portfolio_optimization.ipynb`.
- Keep this summary:
  `GMV-NN` achieves the best annual return, volatility, Sharpe, and Calmar among the six compared strategies.

#### 7.4 Robustness study

- Source file: `mc_results.csv` and `Portfolio.ipynb`.
- Monte Carlo design:
  `200` iterations,
  `30` random stocks per iteration,
  universe size `1338`,
  same train/test period.
- Report these summary numbers:
  mean volatility `12.7555` for NN vs `13.1485` for ARCH;
  mean Sharpe `1.1967` for NN vs `1.1525` for ARCH;
  mean paired volatility gain `+0.3930`;
  mean paired Sharpe gain `+0.0442`.
- Report the Wilcoxon tests:
  volatility `W = 19563.0`, `p = 1.8875e-31`,
  Sharpe `W = 14172.0`, `p = 2.4582e-07`.

#### 7.5 Computational resources and cost

- Verified runtimes in the notebooks:
  50-stock unconstrained precomputation on CPU: about `70.6s` for CNN-GARCH, `1.8s` for GARCH recursion.
  30-stock constrained RIENet notebook on CPU: about `66.1s` for CNN-GARCH, `0.6s` for GARCH recursion.
  100-stock GMV notebook on CPU: about `114.9s` for CNN-GARCH, `4.9s` for GARCH recursion.
  Monte Carlo evaluation: about `42.3 min` on CPU for 200 iterations.
- If no cloud bill exists, explicitly write:
  monetary cost was not tracked because the experiments were run locally on CPU.

### 8. Ethical and societal impact

- Model risk:
  low realized variance in backtests does not guarantee future robustness.
- Market impact and transaction costs:
  limited in the current project, especially for unconstrained and short-selling settings.
- Leverage and shorting:
  can amplify losses and may be unacceptable in real mandates.
- Data bias:
  survivorship, delisting, stale-price effects, and the limits of daily data.
- Explain that the model is a decision-support tool, not a fully autonomous trading system.

### 9. Project organization

- Add a small subsection on who did what:
  literature review,
  data cleaning,
  synthetic pretraining,
  real-data fine-tuning,
  GMV evaluation,
  poster/report writing.
- Add an hours table.
- Add a simple timeline or Gantt-like table by week.

### 10. Conclusion and perspectives

- Be specific:
  your main claim is not that the problem is solved forever,
  but that better marginal volatility estimates and cleaned correlations improve GMV behavior in your experiments.
- Mention the nuanced result:
  RIENet clearly helps in the 50-stock unconstrained setup by lowering volatility,
  but not every cleaned variant dominates every constrained experiment.
- Suggested future work:
  transaction costs and turnover penalties,
  long-only and exposure constraints,
  larger universes,
  joint end-to-end training of volatility and correlation modules,
  stronger benchmarking against shrinkage and risk-parity baselines.

## Figures already available in the repo

- `End-to-end/nn_garch_real_test_diagnostic.png`
- `End-to-end/gmv_unconstrained_results.png`
- `End-to-end/gmv_results.png`
- `mc_wilcoxon_distributions.png`
- `sigma_vs_absr_diagnostic.png`
- `weights_analysis.png`
- `portfolio_comparison.png`

## References that are already verified from the repo

- Markowitz, 1952.
- Engle, 1982.
- Bollerslev, 1986.
- Ledoit and Wolf, 2004.
- Bongiorno et al., 2025, end-to-end large portfolio optimization through covariance cleaning.
- Bongiorno et al., 2025, neural network-driven volatility drag mitigation under aggressive leverage.
