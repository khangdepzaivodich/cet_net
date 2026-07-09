"""
MuscleAU-Net: Full Integrated Model

Connects all 8 stages into a single forward pass:

  Image
    → Stage 1: ViT Backbone → patch_tokens
    → Stage 2: Geometry Branch → geometry_predicates
    → Stage 3: Texture Branch → roi_features, texture_predicates
    → Stage 4: Multi-Modal Repository → 3 projected pools
    → Stage 5: Muscle Query Cross-Attention → muscle_embeddings
    → Stage 6: Muscle Activation → muscle_activations (latent)
    → Stage 7: Muscle Graph Transformer → refined_embeddings
    → Recompute Activations from refined embeddings
    → Stage 8: Neuro-Symbolic Reasoning → AU predictions

Input:  [B, 3, 256, 256] RGB face images
Output: dict with AU predictions, muscle activations, predicates, losses
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from models.backbone import ViTBackbone
from models.geometry_branch import GeometryBranch
from models.texture_branch import TextureBranch
from models.multi_modal_repo import MultiModalRepository
from models.muscle_query_attention import MuscleQueryAttention
from models.muscle_activation import MuscleActivationHead
from models.muscle_graph import MuscleGraphTransformer
from reasoning.predicates import PredicateStore
from reasoning.rules import SymbolicRuleEngine
from reasoning.counterfactual import CounterfactualEngine
from knowledge.muscle_anatomy import get_muscle_adjacency, NUM_MUSCLES


class MuscleAUNet(nn.Module):
    """
    Full MuscleAU-Net model integrating all stages.
    """

    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        backbone_dim=768,
        hidden_dim=256,
        texture_dim=128,
        num_muscles=NUM_MUSCLES,
        num_aus=12,
        au_list=None,
        num_geometry_predicates=8,
        num_texture_predicates=5,
        gat_heads=4,
        gat_layers=2,
        cross_attn_heads=8,
        cross_attn_layers=2,
    ):
        super().__init__()

        if au_list is None:
            au_list = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

        self.num_muscles = num_muscles
        self.num_aus = num_aus

        # Stage 1: ViT Backbone
        self.backbone = ViTBackbone(backbone_name)

        # Stage 2: Geometry Branch
        self.geometry_branch = GeometryBranch(
            num_predicates=num_geometry_predicates,
            embed_dim=hidden_dim,
        )

        # Stage 3: Texture Branch
        self.texture_branch = TextureBranch(
            texture_dim=texture_dim,
            num_predicates=num_texture_predicates,
        )

        # Stage 4: Multi-Modal Repository
        self.multi_modal_repo = MultiModalRepository(
            vit_dim=backbone_dim,
            geo_predicates=num_geometry_predicates,
            tex_dim=texture_dim,
            common_dim=hidden_dim,
        )

        # Stage 5: Muscle Query Cross-Attention
        self.muscle_query_attn = MuscleQueryAttention(
            num_muscles=num_muscles,
            dim=hidden_dim,
            num_heads=cross_attn_heads,
            num_layers=cross_attn_layers,
        )

        # Stage 6: Muscle Activation Head
        self.muscle_activation = MuscleActivationHead(
            embed_dim=hidden_dim,
            num_muscles=num_muscles,
        )

        # Stage 7: Muscle Graph Transformer
        self.muscle_graph = MuscleGraphTransformer(
            dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
        )

        # Second activation head (for refined embeddings)
        self.muscle_activation_refined = MuscleActivationHead(
            embed_dim=hidden_dim,
            num_muscles=num_muscles,
        )

        # Stage 8: Reasoning
        self.rule_engine = SymbolicRuleEngine(num_aus=num_aus, au_list=au_list)
        self.counterfactual_engine = CounterfactualEngine(num_muscles=num_muscles)

        # Register muscle adjacency as buffer
        adj = get_muscle_adjacency(num_muscles)
        self.register_buffer("muscle_adjacency", adj)

    def forward(self, images, landmarks=None):
        """
        Full forward pass through all 8 stages.

        Args:
            images: [B, 3, 256, 256] normalized face images
            landmarks: [B, 468, 3] optional pre-extracted landmarks

        Returns:
            dict with:
                au_preds: [B, num_aus] — final AU predictions
                muscle_activations: [B, 18] — latent muscle activations
                geometry_predicates: [B, 8]
                texture_predicates: [B, 5]
                rule_satisfactions: list of (name, score) tuples
                compatibility_penalties: list of (name, penalty) tuples
                cf_loss: scalar counterfactual loss
                gate_info: dict with attention gate statistics
        """
        # Stage 1: ViT Backbone
        cls_token, patch_tokens = self.backbone(images)  # [B, D], [B, N, D]

        # Stage 2: Geometry Branch
        geo_embedding, geo_predicates = self.geometry_branch(
            images, landmarks
        )  # [B, D_geo], [B, 8]

        # Stage 3: Texture Branch
        roi_features, tex_predicates = self.texture_branch(
            images, landmarks
        )  # [B, 7, D_tex], [B, 5]

        # Stage 4: Multi-Modal Repository
        vit_pool, geo_pool, tex_pool = self.multi_modal_repo(
            patch_tokens, geo_predicates, roi_features
        )  # [B, N, D], [B, 8, D], [B, 7, D]

        # Stage 5: Muscle Query Cross-Attention
        muscle_embeddings, gate_info = self.muscle_query_attn(
            vit_pool, geo_pool, tex_pool
        )  # [B, 18, D]

        # Stage 6: Initial Muscle Activations
        muscle_act_init = self.muscle_activation(muscle_embeddings)  # [B, 18]

        # Stage 7: Muscle Graph Transformer (refine embeddings)
        refined_embeddings = self.muscle_graph(
            muscle_embeddings, self.muscle_adjacency
        )  # [B, 18, D]

        # Recompute activations from refined embeddings
        muscle_activations = self.muscle_activation_refined(refined_embeddings)  # [B, 18]

        # Stage 8: Neuro-Symbolic Reasoning
        preds = PredicateStore(geo_predicates, tex_predicates, muscle_activations)
        au_preds, rule_sats, compat = self.rule_engine(preds, muscle_activations)

        # Counterfactual loss
        cf_loss = self.counterfactual_engine(muscle_activations, self.rule_engine)

        return {
            "au_preds": au_preds,
            "muscle_activations": muscle_activations,
            "muscle_activations_init": muscle_act_init,
            "geometry_predicates": geo_predicates,
            "texture_predicates": tex_predicates,
            "rule_satisfactions": rule_sats,
            "compatibility_penalties": compat,
            "cf_loss": cf_loss,
            "gate_info": gate_info,
        }

    def get_backbone_params(self):
        """Return backbone parameters (for differential LR)."""
        return self.backbone.parameters()

    def get_non_backbone_params(self):
        """Return all non-backbone parameters."""
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        return [p for p in self.parameters() if id(p) not in backbone_ids]
