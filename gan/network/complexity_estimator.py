from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexityEstimator(nn.Module):
    def __init__(
        self,
        n_actions: int,
        max_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        max_score: Optional[float] = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.n_actions = n_actions
        self.input_proj = nn.Linear(n_actions, d_model)
        self.max_score = max_score
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initialize_parameters(self) -> None:
        self.reset_parameters()
        for layer in self.encoder.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, masked_logits: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.softmax(masked_logits, dim=-1)
        x = self.input_proj(probs)
        pos = self.pos_embedding[:, : x.size(1)]
        x = x + pos
        if masks is None:
            step_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        else:
            if masks.dim() == 3:
                step_mask = masks.any(dim=-1)
            elif masks.dim() == 2:
                step_mask = masks.bool()
            else:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")
        step_mask = step_mask.to(dtype=torch.bool)
        key_padding_mask = ~step_mask
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        mask_float = step_mask.float()
        encoded = encoded * mask_float.unsqueeze(-1)
        denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = encoded.sum(dim=1) / denom
        pooled = self.dropout(pooled)
        raw_scores = self.head(pooled).squeeze(-1)
        if self.max_score is None:
            return raw_scores
        bounded = torch.sigmoid(raw_scores)
        return bounded * self.max_score
