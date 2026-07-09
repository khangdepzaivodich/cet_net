"""
Stage 1: Global Appearance Encoder

Swappable Vision Transformer backbone that extracts patch-level tokens
from a face image.

Supported backbones:
  - dinov2_vitb14: DINOv2 ViT-B/14 (best self-supervised features)
  - vit_b_16: torchvision ViT-B/16 (ImageNet pretrained)

Input:  [B, 3, 256, 256] RGB face image
Output:
  - cls_token:    [B, D] global CLS representation
  - patch_tokens: [B, N, D] per-patch features (N depends on backbone)
"""

import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone with swappable architecture.
    Outputs both CLS token and all patch tokens for downstream modules.
    """

    def __init__(self, backbone_name="dinov2_vitb14", freeze_layers=0):
        """
        Args:
            backbone_name: which pretrained ViT to load
            freeze_layers: number of transformer blocks to freeze (0=train all)
        """
        super().__init__()
        self.backbone_name = backbone_name

        if "dinov2" in backbone_name:
            self.model = torch.hub.load("facebookresearch/dinov2", backbone_name)
            self.embed_dim = self.model.embed_dim  # 768 for ViT-B
        elif backbone_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.model = vit
            self.embed_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Optionally freeze early layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers):
        """Freeze the first num_layers transformer blocks."""
        # Freeze patch embedding
        if hasattr(self.model, "patch_embed"):
            for p in self.model.patch_embed.parameters():
                p.requires_grad = False

        # Freeze transformer blocks
        blocks = None
        if hasattr(self.model, "blocks"):
            blocks = self.model.blocks  # DINOv2
        elif hasattr(self.model, "encoder"):
            blocks = self.model.encoder.layers  # torchvision ViT

        if blocks is not None:
            for i, block in enumerate(blocks):
                if i < num_layers:
                    for p in block.parameters():
                        p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            cls_token: [B, D]
            patch_tokens: [B, N, D]
        """
        if "dinov2" in self.backbone_name:
            # DINOv2 models have a forward_features method
            # that returns the intermediate features
            features = self.model.forward_features(x)
            # features dict has 'x_norm_clstoken' and 'x_norm_patchtokens'
            cls_token = features["x_norm_clstoken"]       # [B, D]
            patch_tokens = features["x_norm_patchtokens"]  # [B, N, D]
        elif self.backbone_name == "vit_b_16":
            # torchvision ViT: manually extract tokens
            x = self.model._process_input(x)
            n = x.shape[0]
            # Add CLS token
            cls_tokens = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            x = x + self.model.encoder.pos_embedding
            x = self.model.encoder.ln(self.model.encoder.layers(x))
            cls_token = x[:, 0]       # [B, D]
            patch_tokens = x[:, 1:]   # [B, N, D]

        return cls_token, patch_tokens

    def get_params(self):
        """Return parameters for optimizer (allows differential LR)."""
        return self.model.parameters()
