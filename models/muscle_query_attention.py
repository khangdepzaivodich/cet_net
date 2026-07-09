"""
Stage 5: Muscle Query Cross-Attention

18 learnable muscle queries attend to all three modality pools
to produce one embedding per facial muscle.

Each muscle query performs cross-attention over:
  1. ViT patch tokens (global appearance)
  2. Geometry predicate tokens
  3. ROI texture features

The three resulting vectors are fused via a learned gating mechanism.

Input:
  - vit_pool:  [B, N, D]
  - geo_pool:  [B, 8, D]
  - tex_pool:  [B, 7, D]

Output:
  - muscle_embeddings: [B, 18, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Multi-head cross-attention: queries attend to keys/values from a pool.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries, kv_pool, logic_bias=None):
        """
        Args:
            queries: [B, Q, D] — muscle queries
            kv_pool: [B, S, D] — keys/values from a modality pool
            logic_bias: [B, Q, S] optional attention bias from logic engine

        Returns:
            updated_queries: [B, Q, D]
            attn_weights: [B, Q, S]
        """
        # Cross-attention with optional logic bias
        if logic_bias is not None:
            attn_out, attn_weights = self.attn(
                queries, kv_pool, kv_pool,
                attn_mask=logic_bias,
            )
        else:
            attn_out, attn_weights = self.attn(queries, kv_pool, kv_pool)

        # Residual + norm
        queries = self.norm1(queries + attn_out)

        # FFN + residual + norm
        queries = self.norm2(queries + self.ffn(queries))

        return queries, attn_weights


class MuscleQueryAttention(nn.Module):
    """
    18 learnable muscle queries attend to 3 modality pools,
    with gated fusion of the results.
    """

    def __init__(self, num_muscles=18, dim=256, num_heads=8,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.num_muscles = num_muscles
        self.dim = dim

        # Learnable muscle queries
        self.muscle_queries = nn.Parameter(
            torch.randn(1, num_muscles, dim) * 0.02
        )

        # Separate cross-attention stacks for each modality
        self.vit_attn_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.geo_attn_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.tex_attn_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Gated fusion: combine outputs from 3 modalities
        self.gate_proj = nn.Linear(dim * 3, 3)  # produces 3 gate weights
        self.fusion_norm = nn.LayerNorm(dim)

    def forward(self, vit_pool, geo_pool, tex_pool, logic_bias=None):
        """
        Args:
            vit_pool: [B, N, D]
            geo_pool: [B, 8, D]
            tex_pool: [B, 7, D]
            logic_bias: optional dict with logic biases per modality

        Returns:
            muscle_embeddings: [B, 18, D]
            attn_weights: dict with attention weights per modality
        """
        B = vit_pool.shape[0]

        # Expand muscle queries for batch
        queries = self.muscle_queries.expand(B, -1, -1)  # [B, 18, D]

        # Cross-attend to each modality
        q_vit = queries
        vit_bias = logic_bias["vit"] if logic_bias is not None else None
        for layer in self.vit_attn_layers:
            q_vit, _ = layer(q_vit, vit_pool, logic_bias=vit_bias)

        q_geo = queries
        geo_bias = logic_bias["geo"] if logic_bias is not None else None
        for layer in self.geo_attn_layers:
            q_geo, _ = layer(q_geo, geo_pool, logic_bias=geo_bias)

        q_tex = queries
        tex_bias = logic_bias["tex"] if logic_bias is not None else None
        for layer in self.tex_attn_layers:
            q_tex, _ = layer(q_tex, tex_pool, logic_bias=tex_bias)

        # Gated fusion
        concat = torch.cat([q_vit, q_geo, q_tex], dim=-1)  # [B, 18, 3D]
        gates = F.softmax(self.gate_proj(concat), dim=-1)   # [B, 18, 3]

        fused = (
            gates[:, :, 0:1] * q_vit +
            gates[:, :, 1:2] * q_geo +
            gates[:, :, 2:3] * q_tex
        )  # [B, 18, D]

        muscle_embeddings = self.fusion_norm(fused)

        return muscle_embeddings, {
            "vit_gates": gates[:, :, 0].mean().item(),
            "geo_gates": gates[:, :, 1].mean().item(),
            "tex_gates": gates[:, :, 2].mean().item(),
        }
