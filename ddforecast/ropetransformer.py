"""
2025-06-07 on chatgpt.com
rewrite the model ('simpletransformer') using rotational positional encoding

```
Rotary Positional Embeddings encode position directly into the attention mechanism by rotating the query and key vectors by a function of position — no additive positional encoding is needed.

It works well for continuous and extrapolatable time representations, and it’s especially useful in time series.
```

| Feature                | Benefit                        |
| ---------------------- | ------------------------------ |
| Rotary Positional Emb. | Handles variable/long horizons |
| No fixed sinusoidal PE | Cleaner, extrapolatable        |
| Better generalization  | Especially across sparse time  |
"""

import math
import torch
import torch.nn as nn


def apply_rotary_pos_emb(x, sin, cos):
    # x: [batch, seq_len, n_heads, head_dim]
    x1, x2 = x[...,::2], x[...,1::2]
    x_rotated = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return x_rotated


def build_rope_cache(seq_len, dim, device, base=10000):
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
    dim_idx = torch.arange(0, dim, 2, device=device)  # [dim//2]
    freqs = 1.0 / (base**(dim_idx / dim))
    angles = position * freqs  # [seq_len, dim//2]

    sin = angles.sin().repeat_interleave(2, dim=-1)  # [seq_len, dim]
    cos = angles.cos().repeat_interleave(2, dim=-1)  # [seq_len, dim]
    return sin, cos


class RoPEMultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**(-0.5)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # [B, T, H, D_head]

        sin, cos = build_rope_cache(T, self.head_dim, x.device, rope_base=10000)
        sin, cos = sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]

        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v)  # [B, T, H, D_head]

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class RoPETransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RoPETransformer(nn.Module):

    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=3, dropout=0.1, forecast_horizon=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.transformer_blocks = nn.ModuleList([
            RoPETransformerBlock(model_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(model_dim, forecast_horizon)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.input_proj(x)  # [B, T, model_dim]
        for block in self.transformer_blocks:
            x = block(x)
        x = x[:, -1]  # Take the last time step
        return self.decoder(x)  # [B, forecast_horizon]


class RoPETransformerMasked(nn.Module):
    """RoPE Transformer With Masked Input

    1. Model input = [value, mask] per time step
    2. Use the mask to:
       - Indicate missing values
       - Possibly improve learning by conditioning the model on presence/absence
    3. Ensure masking is respected in attention if needed

    Each time step will be:
    ```py
    input_t = [value_t, mask_t]  # mask = 1 if observed, 0 if missing
    ```
    """

    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=3, dropout=0.1, forecast_horizon=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.transformer_blocks = nn.ModuleList([
            RoPETransformerBlock(model_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(model_dim, forecast_horizon)

    def forward(self, x, input_mask=None):
        # x: [B, T, 2] — includes value and observed flag
        # input_mask: optional [B, T] mask for attention (1 = observed, 0 = missing)

        x = self.input_proj(x)  # [B, T, model_dim]

        for block in self.transformer_blocks:
            x = block(x)  # No masked attention here (can add if needed)

        x = x[:, -1]  # Use last time step's output
        return self.decoder(x)  # [B, forecast_horizon]
