from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    eps: float = 1e-8,
) -> tuple[float, float, float, float]:
    model.train()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_w = 0.0
    total_a = 0.0
    total_b = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        pred_sigma2, omega, alpha, beta = model(x)
        loss = criterion(torch.log(pred_sigma2 + eps), torch.log(y + eps))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_w += omega.detach().mean().item() * bs
        total_a += alpha.detach().mean().item() * bs
        total_b += beta.detach().mean().item() * bs
        n += bs

    return total_loss / n, total_w / n, total_a / n, total_b / n


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 1e-8,
) -> tuple[float, float, float, float]:
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_w = 0.0
    total_a = 0.0
    total_b = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        pred_sigma2, omega, alpha, beta = model(x)
        loss = criterion(torch.log(pred_sigma2 + eps), torch.log(y + eps))

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_w += omega.detach().mean().item() * bs
        total_a += alpha.detach().mean().item() * bs
        total_b += beta.detach().mean().item() * bs
        n += bs

    return total_loss / n, total_w / n, total_a / n, total_b / n


def train_hybrid_garch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    lr: float,
    device: torch.device,
    ckpt_path: str | None = "hybrid_garch_pretrained_synth.pt",
    weight_decay: float = 1e-4,
    scheduler_patience: int = 2,
    print_every: int = 2,
) -> dict[str, list[float]]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )

    best_val = float("inf")
    best_state: dict[str, Any] | None = None

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
        tr_loss, tr_w, tr_a, tr_b = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_w, va_a, va_b = validate_epoch(model, val_loader, device)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)

        if ep == 1 or ep % print_every == 0:
            print(
                f"Epoch {ep:02d} | train={tr_loss:.6f} | val={va_loss:.6f} | "
                f"alpha={va_a:.4f} beta={va_b:.4f}"
            )

    print(f"Best val loss = {best_val:.6f} (log-variance MSE)")

    if ckpt_path is not None and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    elif best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def eval_log_mse(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 1e-8,
) -> float:
    model.eval()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        sigma2_hat, _, _, _ = model(x)
        loss = ((torch.log(sigma2_hat + eps) - torch.log(y + eps)) ** 2).mean()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(1, n)


def arch_baseline_log_mse_per_series(
    returns_mat: torch.Tensor,
    vars_mat: torch.Tensor,
    window_size: int,
    fit_ratio: float = 0.6,
    eps: float = 1e-8,
) -> float:
    from arch import arch_model

    n_series, t_len = returns_mat.shape
    max_start = t_len - window_size - 1
    idxs = np.arange(0, max_start + 1) + window_size

    fit_len = int(fit_ratio * t_len)
    mses = []

    for j in range(n_series):
        r = returns_mat[j].detach().cpu().numpy()
        true_s2 = vars_mat[j].detach().cpu().numpy()

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
        sigma2_hat[:fit_len] = res.conditional_volatility**2

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


@torch.no_grad()
def nn_predict_sigma2_from_window(
    model: torch.nn.Module,
    window_r: np.ndarray,
    mean_train: float,
    std_train: float,
    device: torch.device,
    eps: float = 1e-12,
) -> float:
    """One-step variance prediction from a single return window (real units)."""
    w = (window_r - mean_train) / (std_train + 1e-8)
    xb = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)

    out = model(xb)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    pred = torch.clamp(pred, min=eps).view(-1)[0].item()

    return pred * (std_train + 1e-8) ** 2


@torch.no_grad()
def rolling_forecast_nn(
    model: torch.nn.Module,
    r: np.ndarray,
    window_size: int,
    train_ratio: float = 0.7,
    device: torch.device | str = "cpu",
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
        s2_hat = nn_predict_sigma2_from_window(model, window, mean_train, std_train, device)
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
        s2_hat = float(f.variance.values[-1, 0] / (100**2))

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
