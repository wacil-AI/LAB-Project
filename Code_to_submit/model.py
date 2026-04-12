from __future__ import annotations

import tensorflow as tf


class GroupNorm(tf.keras.layers.Layer):
    """Group Normalization for channels-last tensors [B, L, C]."""

    def __init__(self, num_groups: int, eps: float = 1e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.eps = eps

    def build(self, input_shape):
        C = input_shape[-1]
        self.gamma = self.add_weight(shape=(C,), initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=(C,), initializer="zeros", name="beta")

    def call(self, x):
        # x: [B, L, C]
        shape = tf.shape(x)
        B, L = shape[0], shape[1]
        C = x.shape[-1]
        G = self.num_groups

        x_r = tf.reshape(x, [B, L, G, C // G])
        mean, var = tf.nn.moments(x_r, axes=[1, 3], keepdims=True)
        x_norm = (x_r - mean) / tf.sqrt(var + self.eps)
        x_norm = tf.reshape(x_norm, [B, L, C])
        return x_norm * self.gamma + self.beta


class GarchParamNet(tf.keras.Model):
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

        # Build list first, then assign so Keras tracks all layers via __setattr__
        conv_blocks = []
        for i in range(conv_layers):
            dilation = 2 ** i
            conv_blocks.append(
                tf.keras.layers.Conv1D(
                    filters=hidden,
                    kernel_size=kernel,
                    dilation_rate=dilation,
                    padding="same",
                )
            )
            conv_blocks.append(GroupNorm(num_groups=num_groups))
            conv_blocks.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        self._conv_blocks = conv_blocks

        self.fc_dense1 = tf.keras.layers.Dense(hidden)
        self.fc_act1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.fc_dense2 = tf.keras.layers.Dense(hidden // 2)
        self.fc_act2 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.fc_drop = tf.keras.layers.Dropout(dropout)

        self.head_omega = tf.keras.layers.Dense(1)
        self.head_c = tf.keras.layers.Dense(1)
        self.head_rho = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        """
        Args:
            x: tensor of shape [batch, window, 1]
        Returns:
            omega, alpha, beta: tensors of shape [batch]
        """
        h = x  # [B, W, 1] — channels-last, no transpose needed
        for layer in self._conv_blocks:
            if isinstance(layer, tf.keras.layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)

        h = tf.reduce_mean(h, axis=1)  # global average pooling → [B, hidden]

        h = self.fc_dense1(h)
        h = self.fc_act1(h)
        h = self.fc_dense2(h)
        h = self.fc_act2(h)
        h = self.fc_drop(h, training=training)

        omega = tf.nn.softplus(self.head_omega(h)) + 1e-6
        c = tf.sigmoid(self.head_c(h)) * self.max_persistence
        rho = tf.sigmoid(self.head_rho(h))

        alpha = c * rho
        beta = c * (1.0 - rho)
        return tf.squeeze(omega, axis=-1), tf.squeeze(alpha, axis=-1), tf.squeeze(beta, axis=-1)


class GarchVolLayer(tf.keras.layers.Layer):
    """Differentiable GARCH(1,1) recursion over a return window."""

    def call(self, r_window, omega, alpha, beta):
        r = tf.squeeze(r_window, axis=-1)  # [B, W]
        sigma2 = omega / (1.0 - alpha - beta + 1e-8)

        W = r.shape[1]  # static window size
        for t in range(W):
            sigma2 = omega + alpha * (r[:, t] ** 2) + beta * sigma2

        return tf.maximum(sigma2, 1e-10)


class HybridGarch(tf.keras.Model):
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

    def call(self, r_window, training=False):
        omega, alpha, beta = self.net(r_window, training=training)
        sigma2 = self.vol(r_window, omega, alpha, beta)
        return sigma2, omega, alpha, beta


__all__ = ["GarchParamNet", "GarchVolLayer", "HybridGarch"]
