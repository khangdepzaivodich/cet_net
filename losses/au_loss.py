"""
AU Supervision Loss

Weighted Binary Cross-Entropy for AU prediction.
Handles class imbalance (most AUs are inactive most of the time).
"""

import torch
import torch.nn as nn


class AULoss(nn.Module):
    """
    Weighted BCE loss for multi-label AU prediction.
    Positive samples get higher weight to handle class imbalance.
    """

    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, au_preds, au_labels):
        """
        Args:
            au_preds: [B, num_aus] predicted probabilities in [0, 1]
            au_labels: [B, num_aus] ground truth binary labels

        Returns:
            loss: scalar
        """
        weight = torch.where(
            au_labels == 1,
            torch.tensor(self.pos_weight, device=au_labels.device),
            torch.tensor(1.0, device=au_labels.device),
        )
        loss = nn.functional.binary_cross_entropy(
            au_preds, au_labels, weight=weight
        )
        return loss
