# -*- coding: utf-8 -*-


from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class MultiHeadsAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_int = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        context = x if context is None else context

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda val: rearrange(val, "B T (H E) -> B H T E", H=self.num_heads),
            (q, k, v),
        )

        scaling_dot_prod = (
            torch.einsum("B H I J, B H K J -> B H I K", q, k) * self.scaling
        )

        if attn_mask is not None:
            assert attn_mask.ndim == 2
            # [B, 1, 1, T]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.broadcast_to(scaling_dot_prod.shape)
            reversed_mask = 1.0 - attn_mask
            scaling_dot_prod = scaling_dot_prod.masked_fill(
                reversed_mask.to(torch.bool), 1e-10
            )

        scaling_dot_prod = F.softmax(scaling_dot_prod, dim=-1)
        out_val = torch.einsum("B H I K, B H K J -> B H I J", scaling_dot_prod, v)
        out_val = rearrange(out_val, "B H T E -> B T (H E)")
        return self.dropout(self.out_proj(out_val))
