"""
Counterfactual Consistency Loss

Wrapper around the CounterfactualEngine output.
"""

import torch.nn as nn


class CounterfactualLoss(nn.Module):
    """Direct pass-through — the engine already computes the loss."""

    def forward(self, cf_loss_value):
        """
        Args:
            cf_loss_value: scalar from CounterfactualEngine.forward()

        Returns:
            loss: scalar
        """
        return cf_loss_value
