from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_predictions_analysis(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    n_examples: int = 500,
    true_alpha_mean: float | None = None,
    true_beta_mean: float | None = None,
    save_path: str | None = None,
) -> dict[str, float]:
    """Diagnostic plots for predicted variance and parameters."""
    true_vals = []
    pred_vals = []
    alphas = []
    betas = []

    for x, y in dataset:
        pred_sigma2, _, alpha, beta = model(x, training=False)

        true_vals.extend(y.numpy().tolist())
        pred_vals.extend(pred_sigma2.numpy().tolist())
        alphas.extend(alpha.numpy().tolist())
        betas.extend(beta.numpy().tolist())

        if len(true_vals) >= n_examples:
            break

    true_vals = np.array(true_vals[:n_examples])
    pred_vals = np.array(pred_vals[:n_examples])
    alphas = np.array(alphas[:n_examples])
    betas = np.array(betas[:n_examples])

    residuals = pred_vals - true_vals

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    vmax = max(true_vals.max(), pred_vals.max())
    axes[0, 0].scatter(true_vals, pred_vals, alpha=0.5, s=18)
    axes[0, 0].plot([0, vmax], [0, vmax], "r--", linewidth=1.2, label="ideal")
    axes[0, 0].set_xlabel("True variance")
    axes[0, 0].set_ylabel("Pred variance")
    axes[0, 0].set_title("Predicted vs true variance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(0.0, color="r", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xlabel("Residual (pred - true)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Residual distribution")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(alphas, bins=20, alpha=0.7, edgecolor="black", label="pred alpha")
    if true_alpha_mean is not None:
        axes[1, 0].axvline(true_alpha_mean, color="r", linestyle="--", linewidth=1.2, label="true alpha mean")
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Alpha distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(betas, bins=20, alpha=0.7, edgecolor="black", label="pred beta")
    if true_beta_mean is not None:
        axes[1, 1].axvline(true_beta_mean, color="r", linestyle="--", linewidth=1.2, label="true beta mean")
    axes[1, 1].set_xlabel("beta")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Beta distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    stats = {
        "mse": float(np.mean((pred_vals - true_vals) ** 2)),
        "mae": float(np.mean(np.abs(residuals))),
        "residual_std": float(np.std(residuals)),
        "alpha_mean": float(np.mean(alphas)),
        "beta_mean": float(np.mean(betas)),
        "persistence_mean": float(np.mean(alphas + betas)),
    }

    var_true = np.var(true_vals)
    stats["r2"] = float(1.0 - np.var(residuals) / var_true) if var_true > 0 else float("nan")

    print("MSE:", f"{stats['mse']:.6f}")
    print("MAE:", f"{stats['mae']:.6f}")
    print("R2:", f"{stats['r2']:.4f}")
    print("alpha mean +/- std:", f"{stats['alpha_mean']:.4f} +/- {np.std(alphas):.4f}")
    print("beta mean +/- std:", f"{stats['beta_mean']:.4f} +/- {np.std(betas):.4f}")
    print("alpha + beta mean:", f"{stats['persistence_mean']:.4f}")

    return stats


def rolling_params(
    model: tf.keras.Model,
    r_series: np.ndarray,
    window_size: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict omega, alpha, beta over rolling windows for one series."""
    r_np = np.asarray(r_series, dtype=np.float32)
    # sliding windows: [N, window_size]
    windows = np.lib.stride_tricks.sliding_window_view(r_np, window_size)[:-1]
    x = tf.constant(windows[:, :, np.newaxis], dtype=tf.float32)  # [N, W, 1]

    omegas = []
    alphas = []
    betas = []

    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        omega, alpha, beta = model.net(xb, training=False)
        omegas.append(omega.numpy())
        alphas.append(alpha.numpy())
        betas.append(beta.numpy())

    omega = np.concatenate(omegas)
    alpha = np.concatenate(alphas)
    beta = np.concatenate(betas)
    c = alpha + beta
    return omega, alpha, beta, c


def plot_persistence_over_time(
    model: tf.keras.Model,
    r_series: np.ndarray,
    window_size: int,
    c_max: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot persistence C=alpha+beta over time for one series."""
    _, _, _, c = rolling_params(model, r_series, window_size)
    t = np.arange(window_size, window_size + len(c))

    plt.figure(figsize=(12, 4))
    plt.plot(t, c, label="C = alpha + beta")
    plt.axhline(c_max, linestyle="--", alpha=0.8, label=f"C max ({c_max:.2f})")
    plt.title("Predicted persistence over time")
    plt.xlabel("t")
    plt.ylabel("C")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return t, c


def collect_true_pred_sigma2(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: int | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect variance targets and predictions from a dataset."""
    y_true_list = []
    y_pred_list = []

    for b, batch in enumerate(dataset):
        if max_batches is not None and b >= max_batches:
            break

        xb, yb = batch[0], batch[1]
        out = model(xb, training=False)
        pred = out[0] if isinstance(out, (tuple, list)) else out

        pred = tf.maximum(pred, eps)
        yb = tf.maximum(tf.cast(yb, tf.float32), eps)

        y_true_list.append(yb.numpy().reshape(-1))
        y_pred_list.append(pred.numpy().reshape(-1))

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def plot_true_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    to_volatility: bool = False,
    loglog: bool = False,
    title: str = "Predicted vs True",
) -> dict[str, float]:
    """Scatter plot and quick metrics for true vs predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if to_volatility:
        y_true = np.sqrt(np.maximum(y_true, 1e-12))
        y_pred = np.sqrt(np.maximum(y_pred, 1e-12))

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.25)
    plt.plot([mn, mx], [mn, mx])
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()

    log_mse = float(np.mean((np.log(np.maximum(y_pred, 1e-12)) - np.log(np.maximum(y_true, 1e-12))) ** 2))
    corr_log = float(np.corrcoef(np.log(np.maximum(y_true, 1e-12)), np.log(np.maximum(y_pred, 1e-12)))[0, 1])

    stats = {
        "n_points": int(len(y_true)),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "log_mse": log_mse,
        "corr_log": corr_log,
    }

    print("n_points:", stats["n_points"])
    print("mse:", f"{stats['mse']:.6f}")
    print("log_mse:", f"{stats['log_mse']:.6f}")
    print("corr_log:", f"{stats['corr_log']:.4f}")

    return stats


def plot_training_curves(
    history: dict[str, list[float]],
    test_loss: float | None = None,
    title: str = "Training curves",
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.get("train_loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    if test_loss is not None:
        plt.axhline(test_loss, color="r", linestyle="--", label=f"test={test_loss:.6f}")
    plt.xlabel("Epoch")
    plt.ylabel("Log-variance MSE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


__all__ = [
    "plot_predictions_analysis",
    "rolling_params",
    "plot_persistence_over_time",
    "collect_true_pred_sigma2",
    "plot_true_pred_scatter",
    "plot_training_curves",
]
