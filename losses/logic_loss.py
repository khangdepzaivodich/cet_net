"""
Logic Loss

Penalizes unsatisfied fuzzy rules. Each rule returns a satisfaction
score in [0, 1]. The loss is the mean unsatisfaction (1 - satisfaction).
"""

import torch
import torch.nn as nn


class LogicLoss(nn.Module):
    """Mean unsatisfaction across all fuzzy rules."""

    def forward(self, rule_satisfactions):
        """
        Args:
            rule_satisfactions: list of (name, [B] score) tuples
                                from SymbolicRuleEngine

        Returns:
            loss: scalar — mean (1 - satisfaction) across rules and batch
        """
        if len(rule_satisfactions) == 0:
            return torch.tensor(0.0)

        unsatisfied = []
        for name, score in rule_satisfactions:
            unsatisfied.append(1.0 - score.mean())

        return torch.stack(unsatisfied).mean()
