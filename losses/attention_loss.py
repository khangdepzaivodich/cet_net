"""
Attention Regularization Loss

Encourages the muscle query attention to focus on anatomically
relevant regions (soft regularization, not a hard constraint).
"""

import torch
import torch.nn as nn


class AttentionLoss(nn.Module):
    """
    Entropy-based attention regularization.
    Encourages attention to be focused (low entropy) rather than uniform.
    """

    def forward(self, gate_info):
        """
        Args:
            gate_info: dict with gate statistics from MuscleQueryAttention
                      (not used for gradient, just monitoring)

        Returns:
            loss: scalar (currently returns 0 — placeholder for logic-guided version)
        """
        # For now, this is a placeholder. The logic-guided attention
        # modification will be implemented when we add the feedback loop.
        return torch.tensor(0.0)
