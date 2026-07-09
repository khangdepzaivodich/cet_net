"""
Stage 6: Muscle Activation Estimation

Converts muscle embeddings to activation probabilities via learned MLPs.
These are LATENT variables — no direct supervision exists for muscles.
They are learned jointly through AU loss + logic constraints.

Input:  muscle_embeddings [B, 18, D]
Output: muscle_activations [B, 18] — probabilities in [0, 1]
"""

import torch
import torch.nn as nn


class MuscleActivationHead(nn.Module):
    """
    Converts each muscle embedding into an activation probability.
    Uses a shared 2-layer MLP with sigmoid output.
    """

    def __init__(self, embed_dim=256, hidden_dim=128, num_muscles=18):
        super().__init__()
        self.num_muscles = num_muscles

        # Shared MLP for all muscles
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, muscle_embeddings):
        """
        Args:
            muscle_embeddings: [B, 18, D]

        Returns:
            activations: [B, 18] probabilities in [0, 1]
        """
        # [B, 18, D] → [B, 18, 1] → [B, 18]
        return self.mlp(muscle_embeddings).squeeze(-1)
