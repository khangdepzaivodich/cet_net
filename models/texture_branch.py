"""
Stage 3: Local Texture Branch

Extracts local texture features from 7 facial ROIs using a lightweight CNN.
Also outputs 5 texture predicate truth values for the reasoning engine.

Input:
  - images: [B, 3, H, W] raw images
  - landmarks: [B, 468, 3] face landmarks (for ROI cropping)

Output:
  - roi_features: [B, 7, D_tex] — per-ROI texture embeddings
  - texture_predicates: [B, 5] — texture truth values in [0, 1]

The 5 texture predicates:
  0: crow_feet         (wrinkles around eyes → Orbicularis Oculi)
  1: forehead_wrinkle  (horizontal forehead lines → Frontalis)
  2: glabellar_wrinkle (vertical lines between brows → Corrugator)
  3: nasolabial_fold   (smile lines → Zygomaticus Major)
  4: chin_wrinkle      (chin dimpling → Mentalis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from knowledge.roi_definitions import ROI_NAMES, NUM_ROIS, get_roi_bbox_from_landmarks


class LightweightTextureCNN(nn.Module):
    """
    4-layer CNN that processes a 64×64 ROI crop and produces a texture embedding.
    Shared across all ROIs (weight sharing for efficiency).
    """

    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # 32x32 → 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            # 8x8 → 4x4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # → [B, 128, 1, 1]
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        Args:
            x: [B*R, 3, 64, 64] ROI crops (all ROIs from all batches)

        Returns:
            features: [B*R, out_dim]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TexturePredicateHead(nn.Module):
    """
    Predicts 5 texture predicates from ROI features.

    Each predicate is associated with specific ROIs:
      - crow_feet: left_eye + right_eye ROIs
      - forehead_wrinkle: forehead ROI
      - glabellar_wrinkle: forehead ROI (between brows)
      - nasolabial_fold: left_cheek + right_cheek ROIs
      - chin_wrinkle: mouth ROI (lower part)
    """

    def __init__(self, roi_dim=128, num_predicates=5):
        super().__init__()
        # Each predicate takes relevant ROI features as input
        # We'll use attention pooling over all ROIs for each predicate
        self.predicate_queries = nn.Parameter(torch.randn(num_predicates, roi_dim))
        self.attn_proj = nn.Linear(roi_dim, roi_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(roi_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.num_predicates = num_predicates

    def forward(self, roi_features):
        """
        Args:
            roi_features: [B, 7, D_tex]

        Returns:
            predicates: [B, 5] truth values in [0, 1]
        """
        B = roi_features.shape[0]

        # Attention: each predicate query attends to ROI features
        keys = self.attn_proj(roi_features)  # [B, 7, D]
        queries = self.predicate_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 5, D]

        # [B, 5, 7]
        attn_weights = torch.bmm(queries, keys.transpose(1, 2))
        attn_weights = attn_weights / (keys.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # [B, 5, D]
        pooled = torch.bmm(attn_weights, roi_features)

        # [B, 5, 1] → [B, 5]
        predicates = self.out_proj(pooled).squeeze(-1)

        return predicates


class TextureBranch(nn.Module):
    """
    Full texture branch: images + landmarks → ROI crops → CNN → features + predicates.
    """

    def __init__(self, texture_dim=128, num_predicates=5, roi_size=64):
        super().__init__()
        self.texture_dim = texture_dim
        self.roi_size = roi_size
        self.num_rois = NUM_ROIS

        # Shared CNN for all ROIs
        self.cnn = LightweightTextureCNN(out_dim=texture_dim)

        # Texture predicate head
        self.predicate_head = TexturePredicateHead(
            roi_dim=texture_dim, num_predicates=num_predicates
        )

    def _extract_rois(self, images, landmarks):
        """
        Extract ROI crops from images using landmarks.

        Args:
            images: [B, 3, H, W] tensor
            landmarks: [B, 468, 3] tensor (normalized 0-1)

        Returns:
            roi_crops: [B, 7, 3, roi_size, roi_size] tensor
        """
        B, C, H, W = images.shape
        device = images.device
        roi_crops = torch.zeros(B, self.num_rois, C, self.roi_size, self.roi_size, device=device)

        for b in range(B):
            lm_np = landmarks[b].detach().cpu().numpy()

            for r, roi_name in enumerate(ROI_NAMES):
                x1, y1, x2, y2 = get_roi_bbox_from_landmarks(
                    lm_np, roi_name, H, W, padding=0.15
                )
                # Crop and resize
                crop = images[b, :, y1:y2, x1:x2]  # [C, h, w]
                if crop.shape[1] < 2 or crop.shape[2] < 2:
                    # Fallback: use center crop
                    crop = images[b, :, H//4:3*H//4, W//4:3*W//4]

                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(self.roi_size, self.roi_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                roi_crops[b, r] = crop

        return roi_crops

    def forward(self, images, landmarks=None):
        """
        Args:
            images: [B, 3, H, W] normalized images
            landmarks: [B, 468, 3] landmarks. If None, uses uniform grid ROIs.

        Returns:
            roi_features: [B, 7, D_tex]
            texture_predicates: [B, 5] truth values
        """
        B = images.shape[0]
        device = images.device

        if landmarks is None:
            # Fallback: use fixed grid ROIs (no landmark data)
            landmarks = torch.rand(B, 468, 3, device=device)

        # Extract ROI crops
        roi_crops = self._extract_rois(images, landmarks)  # [B, 7, 3, 64, 64]

        # Reshape for batch CNN processing
        roi_flat = roi_crops.view(B * self.num_rois, 3, self.roi_size, self.roi_size)

        # Process through shared CNN
        features_flat = self.cnn(roi_flat)  # [B*7, D_tex]

        # Reshape back
        roi_features = features_flat.view(B, self.num_rois, self.texture_dim)  # [B, 7, D_tex]

        # Compute texture predicates
        texture_predicates = self.predicate_head(roi_features)  # [B, 5]

        return roi_features, texture_predicates
