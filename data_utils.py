from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GARCHGenerator:
    """Synthetic GARCH(1,1) series generator."""

    def __init__(self, omega: float, alpha: float, beta: float, n_samples: int, seed: int = 42) -> None:
        if not (omega > 0 and alpha >= 0 and beta >= 0 and (alpha + beta) < 1.0):
            raise ValueError("invalid GARCH parameters: need omega>0, alpha>=0, beta>=0, alpha+beta<1")

        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n_samples = int(n_samples)
        self.seed = int(seed)

    def generate_series(self, burn_in: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed)
        sigma2 = self.omega / (1.0 - self.alpha - self.beta)

        returns = []
        variances = []

        for _ in range(self.n_samples + int(burn_in)):
            eps = torch.randn((), generator=g).item()
            r_t = float(np.sqrt(sigma2) * eps)

            returns.append(r_t)
            variances.append(float(sigma2))

            sigma2 = self.omega + self.alpha * (r_t**2) + self.beta * sigma2

        returns_t = torch.tensor(returns[burn_in:], dtype=torch.float32)
        variances_t = torch.tensor(variances[burn_in:], dtype=torch.float32)

        if len(returns_t) != self.n_samples or len(variances_t) != self.n_samples:
            raise RuntimeError("unexpected synthetic series length")

        return returns_t, variances_t


def compute_garch_variance_from_params(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_sigma2: float | None = None,
) -> np.ndarray:
    """Recompute sigma2[t] from returns and GARCH(1,1) parameters."""
    t_len = len(returns)
    sigma2 = np.zeros(t_len, dtype=np.float64)

    if init_sigma2 is None:
        denom = max(1e-8, 1.0 - alpha - beta)
        init_sigma2 = omega / denom

    sigma2[0] = float(init_sigma2)

    for t in range(1, t_len):
        sigma2[t] = omega + alpha * (returns[t - 1] ** 2) + beta * sigma2[t - 1]

    return sigma2


class MultiGARCHDataset(Dataset):
    """Sliding-window dataset built from multiple return series."""

    def __init__(self, returns_mat: torch.Tensor, variances_mat: torch.Tensor, window_size: int, stride: int = 10) -> None:
        if returns_mat.shape != variances_mat.shape:
            raise ValueError("returns_mat and variances_mat must have same shape")

        self.returns = returns_mat
        self.variances = variances_mat
        self.window_size = int(window_size)
        self.stride = int(stride)

        self.n_series, self.t_len = self.returns.shape
        self.max_start = self.t_len - self.window_size - 1
        if self.max_start < 0:
            raise ValueError(
                f"series length={self.t_len} too short for window_size={self.window_size}; need T >= W+1"
            )

        self.windows_per_series = self.max_start // self.stride + 1
        self.n_samples = self.n_series * self.windows_per_series

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_idx = idx // self.windows_per_series
        k = idx % self.windows_per_series

        t = k * self.stride
        x = self.returns[series_idx, t : t + self.window_size].unsqueeze(-1)  # [W, 1]
        y = self.variances[series_idx, t + self.window_size]  # scalar
        return x, y


def sample_params_by_C(
    n_series: int,
    seed: int,
    c_range: tuple[float, float],
    rho_range: tuple[float, float] = (0.05, 0.95),
    vbar_range: tuple[float, float] = (0.5, 5.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample GARCH params through persistence C=alpha+beta and split rho."""
    rng = np.random.default_rng(seed)

    c = rng.uniform(c_range[0], c_range[1], size=n_series)
    rho = rng.uniform(rho_range[0], rho_range[1], size=n_series)
    vbar = rng.uniform(vbar_range[0], vbar_range[1], size=n_series)

    alpha = c * rho
    beta = c * (1.0 - rho)
    omega = vbar * (1.0 - c)

    return omega, alpha, beta, c


def make_synth_dataset(
    omegas: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    n_samples: int = 2000,
    burn_in: int = 200,
    seed0: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns_list = []
    vars_list = []

    for j in range(len(omegas)):
        gen = GARCHGenerator(omegas[j], alphas[j], betas[j], n_samples=n_samples, seed=seed0 + j)
        r, s2 = gen.generate_series(burn_in=burn_in)
        returns_list.append(r)
        vars_list.append(s2)

    returns_mat = torch.stack(returns_list, dim=0)
    vars_mat = torch.stack(vars_list, dim=0)
    return returns_mat, vars_mat


def split_series_indices(
    n_series: int,
    train_split: float = 0.6,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_split <= 0 or val_split < 0 or (train_split + val_split) >= 1.0:
        raise ValueError("need train_split > 0, val_split >= 0 and train_split + val_split < 1")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_series)

    n_tr = int(train_split * n_series)
    n_va = int(val_split * n_series)

    tr_ids = idx[:n_tr]
    va_ids = idx[n_tr : n_tr + n_va]
    te_ids = idx[n_tr + n_va :]
    return tr_ids, va_ids, te_ids


def standardize_returns(returns: torch.Tensor, mean_r: torch.Tensor, std_r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - mean_r) / (std_r + eps)


def scale_variances(variances: torch.Tensor, std_r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return variances / ((std_r + eps) ** 2)


__all__ = [
    "GARCHGenerator",
    "compute_garch_variance_from_params",
    "MultiGARCHDataset",
    "sample_params_by_C",
    "make_synth_dataset",
    "split_series_indices",
    "standardize_returns",
    "scale_variances",
]
