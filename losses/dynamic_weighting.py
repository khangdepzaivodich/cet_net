"""
Dynamic Loss Weighting

Implements Uncertainty Weighting (Kendall et al., 2018) to automatically
learn the optimal weights for the 5 different loss components during training.

Formula: L = sum(L_i / (2 * sigma_i^2) + log(sigma_i))
We parameterize log(sigma_i^2) directly for numerical stability.
"""

import torch
import torch.nn as nn


class DynamicLossWeighting(nn.Module):
    def __init__(self, num_losses=5):
        super().__init__()
        # Initialize log(sigma^2) to 0, which means weights start at 1.0
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses):
        """
        Args:
            losses: list or tuple of 5 scalar loss tensors
                    (au, logic, counterfactual, graph, attention)
        
        Returns:
            total_loss: scalar tensor
        """
        assert len(losses) == len(self.log_vars)
        
        total_loss = 0.0
        for i, loss in enumerate(losses):
            # weight = exp(-log_var) = 1 / sigma^2
            precision = torch.exp(-self.log_vars[i])
            
            # Loss = 0.5 * precision * L_i + 0.5 * log_var
            # We omit the 0.5 for simplicity, it scales the whole loss equally
            total_loss += precision * loss + self.log_vars[i]
            
        return total_loss

    def get_current_weights(self):
        """Returns the effective weights applied to each loss (for logging)."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).tolist()
