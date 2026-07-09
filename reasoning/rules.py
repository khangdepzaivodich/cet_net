"""
Symbolic Rule Base

Implements all differentiable fuzzy logic rules from plan.md:
  1. Geometry → Muscle rules
  2. Texture → Muscle rules
  3. Multi-evidence → Muscle rules
  4. Muscle → AU mapping
  5. Muscle compatibility rules

All rules return differentiable satisfaction scores in [0, 1].
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reasoning.fuzzy_ops import (
    fuzzy_and, fuzzy_imply, fuzzy_antagonist_penalty, fuzzy_synergist_reward,
)
from reasoning.predicates import PredicateStore
from knowledge.muscle_anatomy import (
    get_muscle_to_au_map, get_muscle_compatibility_pairs, MUSCLE_IDX,
)


class SymbolicRuleEngine(nn.Module):
    """
    Evaluates all symbolic rules and produces:
      1. Rule satisfaction scores (for logic loss)
      2. AU predictions (from muscle → AU mapping)
      3. Compatibility penalties (for graph loss)
    """

    def __init__(self, num_aus=12, au_list=None):
        super().__init__()
        if au_list is None:
            au_list = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

        self.num_aus = num_aus

        # Register muscle→AU mapping as buffer (not trainable)
        muscle_au_map = get_muscle_to_au_map(au_list=au_list)  # [18, num_aus]
        self.register_buffer("muscle_au_map", muscle_au_map)

        # Learnable AU bias (allows fine-tuning the mapping)
        self.au_bias = nn.Parameter(torch.zeros(num_aus))

        # Get compatibility pairs
        self.compatibility_pairs = get_muscle_compatibility_pairs()

    def geometry_to_muscle_rules(self, preds):
        """
        Evaluate Geometry → Muscle implication rules.

        Returns list of (rule_name, satisfaction_score) tuples.
        Satisfaction scores are [B] tensors.
        """
        rules = []

        # CornerUp → Zygomaticus Major
        rules.append(("CornerUp→Zygo",
            fuzzy_imply(preds.geo("CornerUp"), preds.mus("Zygomaticus_Major"))))

        # BrowRaised → Frontalis Medialis
        rules.append(("BrowRaised→FrontM",
            fuzzy_imply(preds.geo("BrowRaised"), preds.mus("Frontalis_Medialis"))))

        # BrowsTogether → Corrugator
        rules.append(("BrowsTogether→Corr",
            fuzzy_imply(preds.geo("BrowsTogether"), preds.mus("Corrugator"))))

        # CornerDown → Depressor Anguli Oris
        rules.append(("CornerDown→DAO",
            fuzzy_imply(preds.geo("CornerDown"), preds.mus("Depressor_Anguli_Oris"))))

        # ChinRaised → Mentalis
        rules.append(("ChinRaised→Mentalis",
            fuzzy_imply(preds.geo("ChinRaised"), preds.mus("Mentalis"))))

        # LipSeparated → Orbicularis Oris (relaxation)
        rules.append(("LipSep→OrbOris",
            fuzzy_imply(preds.geo("LipSeparated"), preds.mus("Orbicularis_Oris"))))

        # CheekRaised → Orbicularis Oculi Orbital
        rules.append(("CheekRaised→OrbOcO",
            fuzzy_imply(preds.geo("CheekRaised"), preds.mus("Orbicularis_Oculi_Orbital"))))

        # EyeOpenness → Levator Palpebrae (wide eyes)
        rules.append(("EyeOpen→LevPalp",
            fuzzy_imply(preds.geo("EyeOpenness"), preds.mus("Levator_Palpebrae"))))

        return rules

    def texture_to_muscle_rules(self, preds):
        """Evaluate Texture → Muscle implication rules."""
        rules = []

        # CrowFeet → Orbicularis Oculi
        rules.append(("CrowFeet→OrbOc",
            fuzzy_imply(preds.tex("CrowFeet"), preds.mus("Orbicularis_Oculi_Orbital"))))

        # ForeheadWrinkle → Frontalis
        rules.append(("ForWrinkle→Front",
            fuzzy_imply(preds.tex("ForeheadWrinkle"), preds.mus("Frontalis_Medialis"))))

        # GlabellarWrinkle → Corrugator
        rules.append(("GlabWrinkle→Corr",
            fuzzy_imply(preds.tex("GlabellarWrinkle"), preds.mus("Corrugator"))))

        # NasolabialFold → Zygomaticus Major
        rules.append(("NasoFold→Zygo",
            fuzzy_imply(preds.tex("NasolabialFold"), preds.mus("Zygomaticus_Major"))))

        # ChinWrinkle → Mentalis
        rules.append(("ChinWrinkle→Ment",
            fuzzy_imply(preds.tex("ChinWrinkle"), preds.mus("Mentalis"))))

        return rules

    def multi_evidence_rules(self, preds):
        """Evaluate Multi-evidence → Muscle rules (AND conditions)."""
        rules = []

        # CornerUp AND NasolabialFold → Zygomaticus
        cond = fuzzy_and(preds.geo("CornerUp"), preds.tex("NasolabialFold"))
        rules.append(("CornerUp∧NasoFold→Zygo",
            fuzzy_imply(cond, preds.mus("Zygomaticus_Major"))))

        # EyeClosing AND CrowFeet → Orbicularis
        # (Use 1 - EyeOpenness as proxy for eye closing)
        eye_closing = 1.0 - preds.geo("EyeOpenness")
        cond = fuzzy_and(eye_closing, preds.tex("CrowFeet"))
        rules.append(("EyeClose∧CrowFeet→OrbOc",
            fuzzy_imply(cond, preds.mus("Orbicularis_Oculi_Orbital"))))

        # BrowRaise AND ForeheadWrinkle → Frontalis
        cond = fuzzy_and(preds.geo("BrowRaised"), preds.tex("ForeheadWrinkle"))
        rules.append(("BrowRaise∧ForWrinkle→Front",
            fuzzy_imply(cond, preds.mus("Frontalis_Medialis"))))

        # ChinRaise AND ChinWrinkle → Mentalis
        cond = fuzzy_and(preds.geo("ChinRaised"), preds.tex("ChinWrinkle"))
        rules.append(("ChinRaise∧ChinWrinkle→Ment",
            fuzzy_imply(cond, preds.mus("Mentalis"))))

        return rules

    def predict_aus(self, muscle_activations):
        """
        Predict AU probabilities from muscle activations using the
        anatomical muscle→AU mapping.

        This is a differentiable soft mapping:
          AU_j = sigmoid(sum_i(muscle_i * map[i,j]) + bias_j)

        Args:
            muscle_activations: [B, 18]

        Returns:
            au_predictions: [B, num_aus] probabilities in [0, 1]
        """
        # [B, 18] × [18, num_aus] → [B, num_aus]
        logits = torch.matmul(muscle_activations, self.muscle_au_map) + self.au_bias
        return torch.sigmoid(logits)

    def compute_compatibility(self, muscle_activations):
        """
        Evaluate muscle compatibility rules.

        Returns:
            penalties: list of (pair_name, penalty_score) tuples
        """
        penalties = []
        for m_a, m_b, relation in self.compatibility_pairs:
            a = muscle_activations[:, m_a]
            b = muscle_activations[:, m_b]

            if relation == "antagonist":
                # High penalty if both active
                penalties.append((f"antag_{m_a}_{m_b}", fuzzy_antagonist_penalty(a, b)))
            elif relation == "weak_antagonist":
                # Weaker penalty
                penalties.append((f"wantag_{m_a}_{m_b}", 0.5 * fuzzy_antagonist_penalty(a, b)))
            elif relation == "synergist":
                # Reward for co-activation (penalty = 1 - reward)
                penalties.append((f"syn_{m_a}_{m_b}", 1.0 - fuzzy_synergist_reward(a, b)))

        return penalties

    def forward(self, preds, muscle_activations):
        """
        Evaluate all rules and produce AU predictions.

        Args:
            preds: PredicateStore with all predicate truth values
            muscle_activations: [B, 18]

        Returns:
            au_predictions: [B, num_aus]
            all_rule_satisfactions: list of (name, [B] score) tuples
            compatibility_penalties: list of (name, [B] score) tuples
        """
        # Collect all rule satisfactions
        all_rules = []
        all_rules.extend(self.geometry_to_muscle_rules(preds))
        all_rules.extend(self.texture_to_muscle_rules(preds))
        all_rules.extend(self.multi_evidence_rules(preds))

        # AU predictions
        au_preds = self.predict_aus(muscle_activations)

        # Compatibility
        compat = self.compute_compatibility(muscle_activations)

        return au_preds, all_rules, compat
