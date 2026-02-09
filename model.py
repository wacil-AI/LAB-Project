from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GarchParamNet(nn.Module):
    """Predicts omega, alpha and beta from a return window."""
    DROP = 0.0
    def __init__(
        self,
        hidden: int = 32,
        kernel: int = 3,
        conv_layers: int = 5,
        dropout: float = DROP,
        max_persistence: float = 0.99,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        if kernel % 2 == 0:
            raise ValueError("kernel must be odd for same-padding with dilation")
        if hidden % num_groups != 0:
            raise ValueError("hidden must be divisible by num_groups for GroupNorm")

        self.max_persistence = float(max_persistence)

        layers = []
        in_ch = 1
        for i in range(conv_layers):
            dilation = 2 ** i
            padding = dilation * (kernel - 1) // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=hidden,
                        kernel_size=kernel,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_ch = hidden

        self.conv = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.head_omega = nn.Linear(hidden // 2, 1)
        self.head_c = nn.Linear(hidden // 2, 1)
        self.head_rho = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape [batch, window, 1]
        Returns:
            omega, alpha, beta: tensors of shape [batch]
        """
        if x.ndim != 3 or x.size(-1) != 1:
            raise ValueError(f"expected input [B, W, 1], got {tuple(x.shape)}")

        x = x.transpose(1, 2)  # [B, 1, W]
        h = self.conv(x).mean(dim=2)  # global average pooling
        h = self.fc(h)

        omega = F.softplus(self.head_omega(h)) + 1e-6
        c = torch.sigmoid(self.head_c(h)) * self.max_persistence
        rho = torch.sigmoid(self.head_rho(h))

        alpha = c * rho
        beta = c * (1.0 - rho)
        return omega.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)


class GarchVolLayer(nn.Module):
    """Differentiable GARCH(1,1) recursion over a return window."""

    def forward(
        self,
        r_window: torch.Tensor,
        omega: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if r_window.ndim != 3 or r_window.size(-1) != 1:
            raise ValueError(f"expected r_window [B, W, 1], got {tuple(r_window.shape)}")

        r = r_window.squeeze(-1)  # [B, W]
        sigma2 = omega / (1.0 - alpha - beta + 1e-8)

        for t in range(r.size(1)):
            sigma2 = omega + alpha * (r[:, t] ** 2) + beta * sigma2

        return torch.clamp(sigma2, min=1e-10)


class HybridGarch(nn.Module):
    """Hybrid architecture: NN parameter head + GARCH volatility layer."""

    def __init__(
        self,
        hidden: int = 32,
        kernel: int = 3,
        conv_layers: int = 5,
        dropout: float = 0.0,
        max_persistence: float = 0.99,
    ) -> None:
        super().__init__()
        self.net = GarchParamNet(
            hidden=hidden,
            kernel=kernel,
            conv_layers=conv_layers,
            dropout=dropout,
            max_persistence=max_persistence,
        )
        self.vol = GarchVolLayer()

    def forward(self, r_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        omega, alpha, beta = self.net(r_window)
        sigma2 = self.vol(r_window, omega, alpha, beta)
        return sigma2, omega, alpha, beta


__all__ = ["GarchParamNet", "GarchVolLayer", "HybridGarch"]
