"""
Stage 4: Multi-Modal Repository

Maintains SEPARATE feature pools from the three branches,
projected to a common dimension. Does NOT concatenate.

These become keys/values for the muscle query cross-attention.

Input:
  - patch_tokens: [B, N, D_vit] from ViT backbone
  - geometry_embedding: [B, D_geo] from geometry branch
  - roi_features: [B, 7, D_tex] from texture branch

Output:
  - vit_pool:  [B, N, D_common]   (projected ViT patch tokens)
  - geo_pool:  [B, 8, D_common]   (geometry as 8 predicate tokens)
  - tex_pool:  [B, 7, D_common]   (projected ROI features)
"""

import torch
import torch.nn as nn


class MultiModalRepository(nn.Module):
    """
    Projects features from each modality to a common dimension,
    maintaining them as separate pools for cross-attention.
    """

    def __init__(self, vit_dim=768, geo_predicates=8, tex_dim=128, common_dim=256):
        """
        Args:
            vit_dim: dimension of ViT patch tokens
            geo_predicates: number of geometry predicates (becomes sequence length)
            tex_dim: dimension of texture ROI features
            common_dim: target projection dimension (D_common)
        """
        super().__init__()

        # Project ViT tokens: [B, N, vit_dim] → [B, N, common_dim]
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, common_dim),
            nn.LayerNorm(common_dim),
        )

        # Project geometry: [B, geo_predicates] → [B, geo_predicates, common_dim]
        # Each predicate becomes a separate token
        self.geo_proj = nn.Sequential(
            nn.Linear(1, common_dim),
            nn.LayerNorm(common_dim),
        )

        # Project texture ROIs: [B, 7, tex_dim] → [B, 7, common_dim]
        self.tex_proj = nn.Sequential(
            nn.Linear(tex_dim, common_dim),
            nn.LayerNorm(common_dim),
        )

        self.common_dim = common_dim
        self.geo_predicates = geo_predicates

    def forward(self, patch_tokens, geometry_predicates, roi_features):
        """
        Args:
            patch_tokens: [B, N, D_vit]
            geometry_predicates: [B, 8] predicate values
            roi_features: [B, 7, D_tex]

        Returns:
            vit_pool: [B, N, D_common]
            geo_pool: [B, 8, D_common]
            tex_pool: [B, 7, D_common]
        """
        # ViT tokens
        vit_pool = self.vit_proj(patch_tokens)  # [B, N, D_common]

        # Geometry: expand each predicate value to its own token
        B = geometry_predicates.shape[0]
        geo_expanded = geometry_predicates.unsqueeze(-1)  # [B, 8, 1]
        geo_pool = self.geo_proj(geo_expanded)  # [B, 8, D_common]

        # Texture ROIs
        tex_pool = self.tex_proj(roi_features)  # [B, 7, D_common]

        return vit_pool, geo_pool, tex_pool
