"""
Stage 7: Anatomical Muscle Graph Transformer

Graph Attention Network (GAT) over 18 muscle nodes with anatomical
adjacency edges. Refines muscle embeddings by allowing anatomically
connected muscles to exchange information.

Input:
  - muscle_embeddings: [B, 18, D]
  - adjacency: [18, 18] binary adjacency matrix

Output:
  - refined_embeddings: [B, 18, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Graph Attention layer.
    Each node attends to its neighbors (defined by adjacency matrix)
    using multi-head attention.
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, adj):
        """
        Args:
            x: [B, N, D_in] node features
            adj: [N, N] adjacency matrix (1=connected, 0=not)

        Returns:
            out: [B, N, D_out] updated node features
        """
        B, N, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # Linear transform and reshape for multi-head
        h = self.W(x)                          # [B, N, H*D]
        h = h.view(B, N, H, D)                 # [B, N, H, D]

        # Compute attention scores
        # e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)
        src_scores = (h * self.a_src).sum(dim=-1)  # [B, N, H]
        dst_scores = (h * self.a_dst).sum(dim=-1)  # [B, N, H]

        # [B, N, H, 1] + [B, 1, H, N] → [B, N, H, N]
        attn = src_scores.unsqueeze(-1) + dst_scores.unsqueeze(1).transpose(2, 3)
        attn = self.leaky_relu(attn)  # [B, N, H, N]

        # Mask: only attend to neighbors (adj=1) and self
        mask = (adj + torch.eye(N, device=adj.device)).unsqueeze(0).unsqueeze(2)
        # [1, N, 1, N]
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)      # [B, N, H, N]
        attn = self.dropout(attn)

        # Aggregate neighbor features
        # [B, N, H, N] × [B, N, H, D] → [B, N, H, D]
        h_t = h.permute(0, 2, 1, 3)  # [B, H, N, D]
        attn_t = attn.permute(0, 2, 1, 3)  # [B, H, N, N]
        out = torch.matmul(attn_t, h_t)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
        out = out.view(B, N, H * D)  # [B, N, D_out]

        # Residual + norm (if dimensions match)
        if x.shape[-1] == out.shape[-1]:
            out = self.norm(out + x)
        else:
            out = self.norm(out)

        return out


class MuscleGraphTransformer(nn.Module):
    """
    Multi-layer GAT over the anatomical muscle graph.
    """

    def __init__(self, dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(dim, dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, muscle_embeddings, adjacency):
        """
        Args:
            muscle_embeddings: [B, 18, D]
            adjacency: [18, 18] binary adjacency matrix

        Returns:
            refined_embeddings: [B, 18, D]
        """
        x = muscle_embeddings
        for layer in self.layers:
            x = layer(x, adjacency)
        return x
