"""Self-attention multi-step forecaster.

The encoder treats one timestep as a token carrying all assets' features, so
temporal self-attention mixes information across assets at the representation
level. Forecasting is then done by ``N x H`` learned queries that cross-attend
back onto the encoder memory - the returned attention weights answer "which
past days drove this asset's h-step-ahead prediction", which is what makes the
model auditable rather than a black box.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionForecaster(nn.Module):
    """Encoder-only transformer with per-(asset, horizon) cross-attention queries.

    Input:  ``x`` of shape ``(B, L, N, F)``
    Output: ``(forecast (B, N, H), attention (B, N, H, L) or None)``
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        lookback: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model

        self.input_proj = nn.Linear(n_assets * n_features, d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        self.queries = nn.Parameter(torch.zeros(n_assets * horizon, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch, steps, assets, features = x.shape
        tokens = x.reshape(batch, steps, assets * features)
        hidden = self.input_proj(tokens) + self.time_pos[:, :steps]
        memory = self.norm(self.encoder(hidden))

        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, weights = self.cross_attn(
            queries, memory, memory, need_weights=return_attention, average_attn_weights=False
        )
        forecast = self.head(attended).squeeze(-1).reshape(batch, assets, self.horizon)
        if not return_attention:
            return forecast, None
        attention = weights.mean(dim=1).reshape(batch, assets, self.horizon, steps)
        return forecast, attention
