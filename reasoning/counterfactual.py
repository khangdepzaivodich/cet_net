"""
Counterfactual Learning via do-Interventions

Implements causal interventions on muscles:
  1. For each muscle m: do(m = 0)  (force muscle to inactive)
  2. Re-run muscle→AU mapping with intervened activations
  3. Check that downstream AUs decrease

The counterfactual consistency loss penalizes cases where:
  - A muscle was active (m > threshold)
  - We set do(m=0)
  - But the AU that muscle maps to did NOT decrease

This enforces causal faithfulness: if the model claims a muscle is active,
removing it must actually affect the predicted AUs.

Input:
  - muscle_activations: [B, 18]
  - rule_engine: SymbolicRuleEngine (for predict_aus)

Output:
  - counterfactual_loss: scalar
"""

import torch
import torch.nn as nn


class CounterfactualEngine(nn.Module):
    """
    Performs do-interventions on muscles and computes consistency loss.
    """

    def __init__(self, num_muscles=18, intervention_threshold=0.3):
        super().__init__()
        self.num_muscles = num_muscles
        self.threshold = intervention_threshold

    def forward(self, muscle_activations, rule_engine):
        """
        Args:
            muscle_activations: [B, 18] — original muscle activations
            rule_engine: SymbolicRuleEngine instance (has predict_aus method)

        Returns:
            cf_loss: scalar — counterfactual consistency loss
        """
        B = muscle_activations.shape[0]
        device = muscle_activations.device

        # Original AU predictions
        au_original = rule_engine.predict_aus(muscle_activations)  # [B, num_aus]

        total_loss = torch.tensor(0.0, device=device)
        num_interventions = 0

        for m in range(self.num_muscles):
            # Only intervene on muscles that are sufficiently active
            active_mask = muscle_activations[:, m] > self.threshold  # [B]
            if not active_mask.any():
                continue

            # do(m = 0): create intervened activations
            intervened = muscle_activations.clone()
            intervened[:, m] = 0.0

            # Re-predict AUs with intervention
            au_intervened = rule_engine.predict_aus(intervened)  # [B, num_aus]

            # Check which AUs this muscle maps to
            au_mask = rule_engine.muscle_au_map[m]  # [num_aus] binary

            if au_mask.sum() == 0:
                continue

            # For mapped AUs: original should be > intervened
            # Loss = ReLU(au_intervened - au_original) for mapped AUs
            # (penalize if AU didn't decrease after removing its muscle)
            diff = au_intervened - au_original  # [B, num_aus]
            violation = torch.relu(diff)        # [B, num_aus]

            # Only count violations for active muscles and mapped AUs
            violation = violation * au_mask.unsqueeze(0)  # [B, num_aus]
            violation = violation * active_mask.float().unsqueeze(-1)  # [B, num_aus]

            total_loss = total_loss + violation.mean()
            num_interventions += 1

        if num_interventions > 0:
            total_loss = total_loss / num_interventions

        return total_loss
