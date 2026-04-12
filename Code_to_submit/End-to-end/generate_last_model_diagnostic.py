from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_stocks_WIKI_price import get_cleaned_data
from model import HybridGarch


SEED = 42
FILE_PATH = Path(
    r"C:\Users\karim\OneDrive\Documents\Msc AI\Finance\TP\WIKI_PRICES_212b326a081eacca455e13140d7bb9db.zip"
)
CHECKPOINT = Path("finetune_real_zeroaware_r2min1e8_best.weights.h5")
OUT_PATH = Path("End-to-end/nn_garch_real_test_diagnostic.png")

WINDOW_SIZE = 90
TRAIN_RANGE = (0, 1000)
TEST_RANGE = (1200, 1400)
N_EVAL = 5000
ZERO_RATIO_MAX = 0.20
R2_MIN = np.float32(1e-8)
EPS = np.float32(1e-12)


def build_fixed_eval_filtered(
    values: np.ndarray,
    split_range: tuple[int, int],
    window_size: int,
    n_samples: int,
    seed: int,
    zero_ratio_max: float,
    r2_min: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_idx, end_idx = split_range
    n_time, n_tickers = values.shape
    min_t = max(start_idx, window_size)
    if min_t >= end_idx:
        raise ValueError("Split too short for window_size.")

    rng = np.random.default_rng(seed)
    x_raw = np.empty((n_samples, window_size), dtype=np.float32)
    y_r2 = np.empty((n_samples,), dtype=np.float32)
    t_out = np.empty((n_samples,), dtype=np.int64)
    j_out = np.empty((n_samples,), dtype=np.int64)

    filled = 0
    attempts = 0
    max_attempts = n_samples * 300

    while filled < n_samples:
        t = int(rng.integers(low=min_t, high=end_idx))
        j = int(rng.integers(low=0, high=n_tickers))
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Too many rejected samples. Try lowering N_EVAL or relaxing filters.")

        r_next = np.float32(values[t, j])
        r2 = np.float32(r_next * r_next)
        if r2 < r2_min:
            continue

        w = values[t - window_size : t, j].astype(np.float32)
        if float((w == 0.0).mean()) >= zero_ratio_max:
            continue

        x_raw[filled] = w
        y_r2[filled] = r2
        t_out[filled] = t
        j_out[filled] = j
        filled += 1

    return x_raw, y_r2, t_out, j_out


def predict_nn_variance(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    mean_r: float,
    std_r: float,
    batch_size: int = 2048,
) -> np.ndarray:
    n = x_raw.shape[0]
    preds = np.empty(n, dtype=np.float32)
    scale_out = np.float32(std_r**2)

    for i in range(0, n, batch_size):
        xb = x_raw[i : i + batch_size].copy()
        xb = (xb - np.float32(mean_r)) / np.float32(std_r + 1e-8)
        xb = xb.astype(np.float32)[..., None]
        sigma2_hat, _, _, _ = model(tf.constant(xb), training=False)
        preds[i : i + batch_size] = (
            np.maximum(sigma2_hat.numpy().reshape(-1), EPS).astype(np.float32) * scale_out
        )

    return preds


def compute_binned_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_pred, q)
    edges[0] = min(edges[0], y_pred.min())
    edges[-1] = max(edges[-1], y_pred.max())

    xs: list[float] = []
    ys: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        if hi == edges[-1]:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        if int(mask.sum()) < 20:
            continue
        xs.append(float(np.median(y_true[mask])))
        ys.append(float(np.median(y_pred[mask])))
    return np.asarray(xs), np.asarray(ys)


def make_plot(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.maximum(y_true.astype(np.float64), float(R2_MIN))
    y_pred = np.maximum(y_pred.astype(np.float64), float(EPS))

    corr_level = float(np.corrcoef(y_true, y_pred)[0, 1])
    log_true = np.log(y_true)
    log_pred = np.log(y_pred)
    corr_log = float(np.corrcoef(log_true, log_pred)[0, 1])
    log_mse = float(np.mean((log_pred - log_true) ** 2))

    x_med, y_med = compute_binned_curve(y_true, y_pred, n_bins=10)
    lo = max(float(min(y_true.min(), y_pred.min())), float(R2_MIN))
    hi = float(np.quantile(np.r_[y_true, y_pred], 0.995) * 1.2)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.facecolor": "#fcfbfd",
            "figure.facecolor": "white",
            "grid.color": "#d7d6dc",
            "grid.alpha": 0.45,
            "axes.edgecolor": "#b7b5be",
        }
    )

    fig = plt.figure(figsize=(10.5, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(
        y_true,
        y_pred,
        xscale="log",
        yscale="log",
        gridsize=42,
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0.0,
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#5c5666", label="ideal y = x")
    if len(x_med) > 0:
        ax.plot(
            x_med,
            y_med,
            color="#7a003c",
            linewidth=2.2,
            marker="o",
            markersize=4.8,
            label="median by prediction decile",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Realized $r^2$")
    ax.set_ylabel("Predicted variance")
    ax.set_title("NN-GARCH on filtered real test windows")
    ax.grid(True, which="both")
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("log density")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#d0ced6")

    stats_box = (
        f"n = {len(y_true):,}\n"
        f"corr level = {corr_level:.3f}\n"
        f"corr log = {corr_log:.3f}\n"
        f"log-MSE = {log_mse:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        stats_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#d0ced6", "alpha": 0.95},
    )

    ax2 = fig.add_subplot(gs[0, 1])
    ratio = np.sqrt(y_pred / y_true)
    r_lo, r_hi = np.quantile(ratio, [0.01, 0.99])
    r_lo = max(float(r_lo), 1e-3)
    r_hi = max(float(r_hi), r_lo * 1.1)
    ratio_mid = ratio[(ratio >= r_lo) & (ratio <= r_hi)]
    bins = np.geomspace(r_lo, r_hi, 28)
    ax2.hist(ratio_mid, bins=bins, color="#3c7db1", alpha=0.85, edgecolor="white")
    med = float(np.median(ratio))
    ax2.axvline(1.0, color="#5c5666", linestyle="--", linewidth=1.4, label="perfect")
    ax2.axvline(med, color="#7a003c", linewidth=2.2, label=f"median = {med:.2f}")
    ax2.set_xscale("log")
    ax2.set_xlim(r_lo, r_hi)
    ax2.set_title("Volatility ratio distribution (central 98%)")
    ax2.set_xlabel(r"$\sqrt{\hat{\sigma}^2 / r^2}$")
    ax2.set_ylabel("Count")
    ax2.grid(True, which="both", axis="x")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#d0ced6")

    fig.suptitle(
        "Last fine-tuned zero-aware CNN-GARCH model",
        fontsize=18,
        fontweight="bold",
        color="#2f2a35",
    )
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_eval": float(len(y_true)),
        "corr_level": corr_level,
        "corr_log": corr_log,
        "log_mse": log_mse,
    }


def main() -> None:
    if not FILE_PATH.exists():
        raise FileNotFoundError(FILE_PATH)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    returns_df = get_cleaned_data(str(FILE_PATH))
    returns_values = returns_df.to_numpy(dtype=np.float32)
    if returns_values.shape[0] < TEST_RANGE[1]:
        raise ValueError(f"Not enough rows for TEST_RANGE={TEST_RANGE} with T={returns_values.shape[0]}")

    train_block = returns_values[TRAIN_RANGE[0] : TRAIN_RANGE[1], :]
    mean_train = float(train_block.mean())
    std_train = float(train_block.std() + 1e-8)

    x_raw, y_true, _, _ = build_fixed_eval_filtered(
        returns_values,
        TEST_RANGE,
        WINDOW_SIZE,
        N_EVAL,
        seed=SEED + 2,
        zero_ratio_max=ZERO_RATIO_MAX,
        r2_min=R2_MIN,
    )

    model = HybridGarch(hidden=32, kernel=3, conv_layers=5)
    _ = model(tf.zeros((1, WINDOW_SIZE, 1), dtype=tf.float32), training=False)
    model.load_weights(str(CHECKPOINT))

    y_pred = predict_nn_variance(model, x_raw, mean_train, std_train)
    stats = make_plot(y_true, y_pred)

    print("saved", OUT_PATH)
    for k, v in stats.items():
        if k == "n_eval":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
