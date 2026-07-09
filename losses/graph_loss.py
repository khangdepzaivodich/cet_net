"""
Graph Compatibility Loss

Penalizes muscle compatibility violations from the anatomical graph.
"""

import torch
import torch.nn as nn


class GraphLoss(nn.Module):
    """Mean compatibility penalty across all muscle pairs."""

    def forward(self, compatibility_penalties):
        """
        Args:
            compatibility_penalties: list of (name, [B] penalty) tuples
                                     from SymbolicRuleEngine.compute_compatibility

        Returns:
            loss: scalar
        """
        if len(compatibility_penalties) == 0:
            return torch.tensor(0.0)

        penalties = []
        for name, penalty in compatibility_penalties:
            penalties.append(penalty.mean())

        return torch.stack(penalties).mean()
