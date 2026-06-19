"""
AU Evidence Field Module
Extracts per-AU evidence, uncertainty, and spatial attention masks.

Input:  [B, 2048, 7, 7]  (dense feature map from backbone)
Outputs:
  - au_features:  [B, K, 256]  per-AU feature vectors
  - beliefs_init: [B, K]       initial AU beliefs (0 to 1)
  - uncertainty:  [B, K]       uncertainty per AU (0 to 1)
  - masks:        [B, K, 7, 7] spatial attention masks

Where K = num_aus (default 12).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUEvidenceField(nn.Module):
    def __init__(self, in_channels=2048, hidden_dim=256, num_aus=12, spatial_size=7):
        super().__init__()
        self.num_aus = num_aus
        self.spatial_size = spatial_size

        # Step 1: Reduce channel dimension from 2048 -> 256
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Step 2: Predict spatial attention masks (one per AU)
        # Each AU gets its own 7x7 heatmap showing where to look
        self.mask_conv = nn.Conv2d(hidden_dim, num_aus, kernel_size=1)

        # Step 3: From each AU's pooled feature vector, predict evidence & uncertainty
        # Shared first layer then separate heads
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        """
        Args:
            features: [B, 2048, 7, 7] from backbone

        Returns:
            au_features:  [B, num_aus, 256]
            beliefs_init: [B, num_aus]
            uncertainty:  [B, num_aus]
            masks:        [B, num_aus, 7, 7]
        """
        B = features.size(0)

        # Step 1: Reduce channels [B, 2048, 7, 7] -> [B, 256, 7, 7]
        feat = self.channel_reduce(features)

        # Step 2: Generate spatial masks [B, 256, 7, 7] -> [B, num_aus, 7, 7]
        mask_logits = self.mask_conv(feat)  # [B, num_aus, 7, 7]
        # Spatial softmax: normalize across the 7x7 grid so each mask sums to 1
        masks = mask_logits.view(B, self.num_aus, -1)  # [B, num_aus, 49]
        masks = F.softmax(masks, dim=-1)  # sum across 49 positions = 1
        masks = masks.view(B, self.num_aus, self.spatial_size, self.spatial_size)

        # Step 3: Masked pooling to get per-AU feature vectors
        # feat:  [B, 256, 7, 7]  -> expand to [B, 1, 256, 7, 7]
        # masks: [B, num_aus, 7, 7] -> expand to [B, num_aus, 1, 7, 7]
        feat_expanded = feat.unsqueeze(1)  # [B, 1, 256, 7, 7]
        masks_expanded = masks.unsqueeze(2)  # [B, num_aus, 1, 7, 7]

        # Multiply and sum over spatial dimensions (7x7)
        # [B, num_aus, 256, 7, 7] -> sum over last two dims -> [B, num_aus, 256]
        au_features = (feat_expanded * masks_expanded).sum(dim=(-2, -1))

        # Step 4: Predict evidence and uncertainty for each AU
        # Reshape to process all AUs at once
        au_flat = au_features.view(B * self.num_aus, -1)  # [B*K, 256]

        evidence = self.evidence_head(au_flat)  # [B*K, 1]
        evidence = evidence.view(B, self.num_aus)  # [B, K]
        beliefs_init = torch.sigmoid(evidence)  # initial beliefs in [0, 1]

        unc = self.uncertainty_head(au_flat)  # [B*K, 1]
        unc = unc.view(B, self.num_aus)  # [B, K]
        uncertainty = torch.sigmoid(unc)  # uncertainty in [0, 1]

        return au_features, beliefs_init, uncertainty, masks
