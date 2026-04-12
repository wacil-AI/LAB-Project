from __future__ import annotations

import os
from typing import Any

import numpy as np
import tensorflow as tf


class _ReduceLROnPlateau:
    """Minimal ReduceLROnPlateau scheduler for a TF optimizer."""

    def __init__(self, optimizer, mode: str = "min", factor: float = 0.5, patience: int = 2) -> None:
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self._wait = 0
        self._best = float("inf") if mode == "min" else float("-inf")
        self._mode = mode

    def step(self, metric: float) -> None:
        improved = (self._mode == "min" and metric < self._best) or (
            self._mode == "max" and metric > self._best
        )
        if improved:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                old_lr = float(self.optimizer.learning_rate)
                self.optimizer.learning_rate.assign(old_lr * self.factor)
                self._wait = 0


def train_epoch(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[float, float, float, float]:
    mse_fn = tf.keras.losses.MeanSquaredError()

    total_loss = 0.0
    total_w = 0.0
    total_a = 0.0
    total_b = 0.0
    n = 0

    for x, y in dataset:
        y = tf.cast(y, tf.float32)

        with tf.GradientTape() as tape:
            pred_sigma2, omega, alpha, beta = model(x, training=True)
            loss = mse_fn(tf.math.log(y + eps), tf.math.log(pred_sigma2 + eps))

        grads = tape.gradient(loss, model.trainable_variables)
        # Apply L2 weight decay manually (equivalent to PyTorch AdamW weight_decay)
        if weight_decay > 0:
            grads = [g + weight_decay * v if g is not None else g
                     for g, v in zip(grads, model.trainable_variables)]
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bs = tf.shape(x)[0].numpy()
        total_loss += float(loss) * bs
        total_w += float(tf.reduce_mean(omega)) * bs
        total_a += float(tf.reduce_mean(alpha)) * bs
        total_b += float(tf.reduce_mean(beta)) * bs
        n += bs

    return total_loss / n, total_w / n, total_a / n, total_b / n


def validate_epoch(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    eps: float = 1e-8,
) -> tuple[float, float, float, float]:
    mse_fn = tf.keras.losses.MeanSquaredError()

    total_loss = 0.0
    total_w = 0.0
    total_a = 0.0
    total_b = 0.0
    n = 0

    for x, y in dataset:
        y = tf.cast(y, tf.float32)

        pred_sigma2, omega, alpha, beta = model(x, training=False)
        loss = mse_fn(tf.math.log(y + eps), tf.math.log(pred_sigma2 + eps))

        bs = tf.shape(x)[0].numpy()
        total_loss += float(loss) * bs
        total_w += float(tf.reduce_mean(omega)) * bs
        total_a += float(tf.reduce_mean(alpha)) * bs
        total_b += float(tf.reduce_mean(beta)) * bs
        n += bs

    return total_loss / n, total_w / n, total_a / n, total_b / n


def train_hybrid_garch(
    model: tf.keras.Model,
    train_loader: tf.data.Dataset,
    val_loader: tf.data.Dataset,
    n_epochs: int,
    lr: float,
    ckpt_path: str | None = "hybrid_garch_pretrained_synth.weights.h5",
    weight_decay: float = 1e-4,
    scheduler_patience: int = 2,
    print_every: int = 2,
) -> dict[str, list[float]]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    scheduler = _ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=scheduler_patience)

    best_val = float("inf")
    best_weights: list[Any] | None = None

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_omega": [],
        "train_alpha": [],
        "train_beta": [],
        "val_omega": [],
        "val_alpha": [],
        "val_beta": [],
    }

    for ep in range(1, n_epochs + 1):
        tr_loss, tr_w, tr_a, tr_b = train_epoch(model, train_loader, optimizer, weight_decay=weight_decay)
        va_loss, va_w, va_a, va_b = validate_epoch(model, val_loader)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_omega"].append(tr_w)
        history["train_alpha"].append(tr_a)
        history["train_beta"].append(tr_b)
        history["val_omega"].append(va_w)
        history["val_alpha"].append(va_a)
        history["val_beta"].append(va_b)

        if va_loss < best_val:
            best_val = va_loss
            best_weights = [w.numpy().copy() for w in model.weights]
            if ckpt_path is not None:
                model.save_weights(ckpt_path)

        if ep == 1 or ep % print_every == 0:
            print(
                f"Epoch {ep:02d} | train={tr_loss:.6f} | val={va_loss:.6f} | "
                f"alpha={va_a:.4f} beta={va_b:.4f}"
            )

    print(f"Best val loss = {best_val:.6f} (log-variance MSE)")

    if ckpt_path is not None and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
    elif best_weights is not None:
        model.set_weights(best_weights)

    return history


def eval_log_mse(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    eps: float = 1e-8,
) -> float:
    total = 0.0
    n = 0

    for x, y in dataset:
        y = tf.cast(y, tf.float32)
        sigma2_hat, _, _, _ = model(x, training=False)
        loss = tf.reduce_mean((tf.math.log(sigma2_hat + eps) - tf.math.log(y + eps)) ** 2)

        bs = tf.shape(x)[0].numpy()
        total += float(loss) * bs
        n += bs

    return total / max(1, n)


def arch_baseline_log_mse_per_series(
    returns_mat,
    vars_mat,
    window_size: int,
    fit_ratio: float = 0.6,
    eps: float = 1e-8,
) -> float:
    from arch import arch_model

    returns_mat = np.asarray(returns_mat)
    vars_mat = np.asarray(vars_mat)

    n_series, t_len = returns_mat.shape
    max_start = t_len - window_size - 1
    idxs = np.arange(0, max_start + 1) + window_size

    fit_len = int(fit_ratio * t_len)
    mses = []

    for j in range(n_series):
        r = returns_mat[j]
        true_s2 = vars_mat[j]

        am = arch_model(
            r[:fit_len],
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="normal",
            rescale=False,
        )
        res = am.fit(disp="off")

        omega = res.params["omega"]
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]

        sigma2_hat = np.zeros(t_len, dtype=np.float64)
        sigma2_hat[:fit_len] = res.conditional_volatility ** 2

        s2 = sigma2_hat[fit_len - 1]
        for t in range(fit_len, t_len):
            s2 = omega + alpha * (r[t - 1] ** 2) + beta * s2
            sigma2_hat[t] = s2

        mse_log = np.mean((np.log(sigma2_hat[idxs] + eps) - np.log(true_s2[idxs] + eps)) ** 2)
        mses.append(mse_log)

    return float(np.mean(mses))


def qlike(r2: np.ndarray, s2_hat: np.ndarray, eps: float = 1e-12) -> float:
    s2_hat = np.maximum(s2_hat, eps)
    return float(np.mean(r2 / s2_hat + np.log(s2_hat)))


def log_mse_on_r2(r2: np.ndarray, s2_hat: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean((np.log(np.maximum(r2, eps)) - np.log(np.maximum(s2_hat, eps))) ** 2))


def nn_predict_sigma2_from_window(
    model: tf.keras.Model,
    window_r: np.ndarray,
    mean_train: float,
    std_train: float,
    eps: float = 1e-12,
) -> float:
    """One-step variance prediction from a single return window (real units)."""
    w = (window_r - mean_train) / (std_train + 1e-8)
    xb = tf.constant(w[np.newaxis, :, np.newaxis], dtype=tf.float32)  # [1, W, 1]

    out = model(xb, training=False)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    pred = float(tf.maximum(pred, eps).numpy().reshape(-1)[0])

    return pred * (std_train + 1e-8) ** 2


def rolling_forecast_nn(
    model: tf.keras.Model,
    r: np.ndarray,
    window_size: int,
    train_ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r)
    n = len(r)
    split = int(train_ratio * n)

    r_train = r[:split]
    mean_train = r_train.mean()
    std_train = r_train.std(ddof=0)

    preds = []
    r2_real = []
    idxs = []

    for t in range(split, n):
        if t - window_size < 0:
            continue
        window = r[t - window_size : t]
        s2_hat = nn_predict_sigma2_from_window(model, window, mean_train, std_train)
        preds.append(s2_hat)
        r2_real.append(r[t] ** 2)
        idxs.append(t)

    return np.array(preds), np.array(r2_real), np.array(idxs)


def rolling_forecast_arch(
    r: np.ndarray,
    window_size: int,
    train_ratio: float = 0.7,
    refit_every: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from arch import arch_model

    r = np.asarray(r)
    n = len(r)
    split = int(train_ratio * n)

    preds = []
    r2_real = []
    idxs = []

    last_fit_t = None
    last_res = None

    for t in range(split, n):
        if t - window_size < 0:
            continue

        if last_fit_t is None or (t - last_fit_t) >= refit_every:
            window = r[t - window_size : t]
            am = arch_model(window * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            last_res = am.fit(disp="off")
            last_fit_t = t

        f = last_res.forecast(horizon=1, reindex=False)
        s2_hat = float(f.variance.values[-1, 0] / (100 ** 2))

        preds.append(s2_hat)
        r2_real.append(r[t] ** 2)
        idxs.append(t)

    return np.array(preds), np.array(r2_real), np.array(idxs)


__all__ = [
    "train_epoch",
    "validate_epoch",
    "train_hybrid_garch",
    "eval_log_mse",
    "arch_baseline_log_mse_per_series",
    "qlike",
    "log_mse_on_r2",
    "nn_predict_sigma2_from_window",
    "rolling_forecast_nn",
    "rolling_forecast_arch",
]
