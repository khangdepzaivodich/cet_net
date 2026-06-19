"""
CET-Net Loss Functions
Four loss terms that jointly train the model.

1. L_AU:   Binary cross-entropy on AU beliefs vs AU ground truth
2. L_expr: Cross-entropy on expression predictions vs expression labels
3. L_rule: Uncertainty-gated rule violation penalty
4. L_cf:   Counterfactual consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge.facs_rules import get_rules


class CETNetLoss(nn.Module):
    def __init__(
        self,
        lambda_au=1.0,
        lambda_expr=1.0,
        lambda_rule=0.5,
        lambda_cf=0.3,
        num_aus=12,
    ):
        super().__init__()
        self.lambda_au = lambda_au
        self.lambda_expr = lambda_expr
        self.lambda_rule = lambda_rule
        self.lambda_cf = lambda_cf
        self.num_aus = num_aus

        self.rules = get_rules()

    def au_loss(self, beliefs, au_labels):
        """
        Binary cross-entropy between predicted AU beliefs and ground truth.

        Args:
            beliefs:   [B, K] predicted AU beliefs (after sigmoid, values 0-1)
            au_labels: [B, K] ground truth AU labels (binary 0 or 1)

        Returns:
            scalar loss
        """
        return F.binary_cross_entropy(beliefs, au_labels.float(), reduction="mean")

    def expr_loss(self, expr_logits, expr_labels):
        """
        Cross-entropy between predicted expression logits and labels.

        Args:
            expr_logits: [B, C] raw logits (before softmax)
            expr_labels: [B] integer class labels

        Returns:
            scalar loss
        """
        return F.cross_entropy(expr_logits, expr_labels, reduction="mean")

    def rule_loss(self, beliefs, uncertainty):
        """
        Uncertainty-gated rule violation penalty.
        Rules fire more strongly when the model is uncertain.

        For each rule (cond_au, expected_au, direction):
          - If direction = +1: violation = max(0, cond_belief - expected_belief)
            (if cond AU is active, expected AU should be too)
          - If direction = -1: violation = cond_belief * expected_belief
            (if cond AU is active, expected AU should NOT be)
          - Gate: gamma = sigmoid(violation + uncertainty_of_expected_au)

        Args:
            beliefs:     [B, K] final AU beliefs
            uncertainty: [B, K] uncertainty scores

        Returns:
            scalar loss
        """
        total_violation = 0.0
        num_rules = len(self.rules)

        for cond_idx, expected_idx, direction in self.rules:
            cond_belief = beliefs[:, cond_idx]       # [B]
            expected_belief = beliefs[:, expected_idx]  # [B]
            unc = uncertainty[:, expected_idx]         # [B]

            if direction > 0:
                # Co-occurrence rule: if cond is high, expected should be high
                violation = F.relu(cond_belief - expected_belief)  # [B]
            else:
                # Mutual exclusion rule: both should not be high
                violation = cond_belief * expected_belief  # [B]

            # Gate: opens wider when uncertain or violating
            gamma = torch.sigmoid(violation + unc)  # [B]

            # Gated violation
            gated_violation = (violation * gamma).mean()
            total_violation = total_violation + gated_violation

        if num_rules > 0:
            total_violation = total_violation / num_rules

        return total_violation

    def counterfactual_loss(self, beliefs_final, expression_head):
        """
        Counterfactual consistency loss.
        If we suppress a random AU, the expression prediction should change,
        and unrelated AUs should stay stable.

        Args:
            beliefs_final:  [B, K] final AU beliefs
            expression_head: the expression head module (to re-run predictions)

        Returns:
            scalar loss
        """
        B, K = beliefs_final.shape

        # Pick a random AU to suppress for each sample
        random_au = torch.randint(0, K, (B,), device=beliefs_final.device)  # [B]

        # Create counterfactual beliefs: set the chosen AU to 0
        cf_beliefs = beliefs_final.clone()
        cf_beliefs[torch.arange(B, device=beliefs_final.device), random_au] = 0.0

        # Get original and counterfactual expression predictions
        with torch.no_grad():
            orig_probs, _ = expression_head(beliefs_final)
        cf_probs, _ = expression_head(cf_beliefs)

        # The expression distribution SHOULD change when we remove an AU
        # We want a minimum divergence between original and counterfactual
        # Use negative KL divergence as loss (encourage change)
        # But clamp to avoid instability
        kl = F.kl_div(
            cf_probs.log().clamp(min=-10),
            orig_probs,
            reduction="batchmean",
            log_target=False,
        )
        # We want some change (kl > 0), penalize if kl is too small
        change_loss = F.relu(0.1 - kl)

        # Unrelated AUs should remain stable
        # Create a mask: 1 for all AUs except the suppressed one
        stability_mask = torch.ones(B, K, device=beliefs_final.device)
        stability_mask[torch.arange(B, device=beliefs_final.device), random_au] = 0.0

        # MSE on the non-suppressed AUs (they should not change)
        # Since we only zeroed one AU, the others in cf_beliefs are the same
        # This term becomes relevant if we re-run the factor graph (future extension)
        stability_loss = (
            (cf_beliefs - beliefs_final).pow(2) * stability_mask
        ).mean()

        return change_loss + stability_loss

    def forward(self, model_output, au_labels, expr_labels, expression_head):
        """
        Compute total loss.

        Args:
            model_output: dict from CETNet.forward()
            au_labels:    [B, K] ground truth AU labels
            expr_labels:  [B] ground truth expression labels
            expression_head: the expression head module (for counterfactual loss)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        beliefs_final = model_output["beliefs_final"]
        expr_logits = model_output["expr_logits"]
        uncertainty = model_output["uncertainty"]

        l_au = self.au_loss(beliefs_final, au_labels)
        l_expr = self.expr_loss(expr_logits, expr_labels)
        l_rule = self.rule_loss(beliefs_final, uncertainty)
        l_cf = self.counterfactual_loss(beliefs_final, expression_head)

        total = (
            self.lambda_au * l_au
            + self.lambda_expr * l_expr
            + self.lambda_rule * l_rule
            + self.lambda_cf * l_cf
        )

        loss_dict = {
            "total": total.item(),
            "au": l_au.item(),
            "expr": l_expr.item(),
            "rule": l_rule.item(),
            "cf": l_cf.item(),
        }

        return total, loss_dict
