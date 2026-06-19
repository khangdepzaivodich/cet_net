"""
CET-Net: Causal Evidence Transport Network
Full model combining all modules.

Pipeline:
  Image -> Backbone -> AU Evidence Field -> Causal Factor Graph -> Expression Head

Input:  [B, 3, 224, 224]
Outputs:
  - expr_probs:    [B, 7]       expression probabilities
  - expr_logits:   [B, 7]       expression logits (for loss)
  - beliefs_init:  [B, 12]      initial AU beliefs (before graph)
  - beliefs_final: [B, 12]      final AU beliefs (after graph)
  - uncertainty:   [B, 12]      AU uncertainty scores
  - masks:         [B, 12, 7, 7] spatial attention masks
"""

import sys
import os

import torch
import torch.nn as nn

# Add project root to path so we can import knowledge module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .backbone import Backbone
from .au_evidence import AUEvidenceField
from .factor_graph import CausalFactorGraph
from .expression_head import ExpressionHead
from knowledge.facs_rules import get_au_adjacency


class CETNet(nn.Module):
    def __init__(
        self,
        num_aus=12,
        num_expressions=7,
        backbone_feat_dim=2048,
        hidden_dim=256,
        spatial_size=7,
        gnn_layers=2,
        pretrained_backbone=True,
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_expressions = num_expressions

        # Phase 1: Backbone
        self.backbone = Backbone(pretrained=pretrained_backbone)

        # Phase 2: AU Evidence Field
        self.au_evidence = AUEvidenceField(
            in_channels=backbone_feat_dim,
            hidden_dim=hidden_dim,
            num_aus=num_aus,
            spatial_size=spatial_size,
        )

        # Phase 3: Causal Factor Graph
        self.factor_graph = CausalFactorGraph(
            hidden_dim=hidden_dim,
            num_aus=num_aus,
            num_layers=gnn_layers,
        )

        # Phase 4: Expression Head
        self.expression_head = ExpressionHead(
            num_aus=num_aus,
            num_expressions=num_expressions,
        )

        # Register the FACS adjacency matrix as a buffer (not a parameter)
        # This is a fixed prior, not learned
        adjacency = get_au_adjacency(num_aus)
        self.register_buffer("adjacency", adjacency)

    def forward(self, x, mode="standard", au_descriptors=None):
        """
        Args:
            x: [B, 3, 224, 224] batch of face images
            mode: "standard", "zsl", or "generalized" (passed to expression head)
            au_descriptors: [C_unseen, K] AU descriptors for unseen classes
                           (required for "zsl" and "generalized" modes)

        Returns:
            dict with keys:
                expr_probs:    [B, num_expressions] (or [B, C_unseen] in zsl mode)
                expr_logits:   [B, num_expressions] (or [B, C_unseen] in zsl mode)
                beliefs_init:  [B, num_aus]
                beliefs_final: [B, num_aus]
                uncertainty:   [B, num_aus]
                masks:         [B, num_aus, spatial_size, spatial_size]
        """
        # Phase 1: Extract features
        features = self.backbone(x)  # [B, 2048, 7, 7]

        # Phase 2: Get AU evidence
        au_features, beliefs_init, uncertainty, masks = self.au_evidence(features)

        # Phase 3: Belief transport through factor graph
        beliefs_final = self.factor_graph(
            au_features, beliefs_init, uncertainty, self.adjacency
        )

        # Phase 4: Compose expression from final AU beliefs
        expr_probs, expr_logits = self.expression_head(
            beliefs_final, mode=mode, au_descriptors=au_descriptors
        )

        return {
            "expr_probs": expr_probs,
            "expr_logits": expr_logits,
            "beliefs_init": beliefs_init,
            "beliefs_final": beliefs_final,
            "uncertainty": uncertainty,
            "masks": masks,
        }

    def get_backbone_params(self):
        """Returns backbone parameters (for lower learning rate)."""
        return self.backbone.parameters()

    def get_non_backbone_params(self):
        """Returns all parameters except backbone (for normal learning rate)."""
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        return [p for p in self.parameters() if id(p) not in backbone_ids]
