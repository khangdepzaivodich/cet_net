"""
Backbone Feature Extractor
Uses ResNet-50 up to conv5_x to produce a dense feature map.

Input:  [B, 3, 224, 224]  (batch of RGB images)
Output: [B, 2048, 7, 7]   (dense feature map)
"""

import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Keep everything except avgpool and fc
        # ResNet-50 layers: conv1, bn1, relu, maxpool, layer1..4
        self.features = nn.Sequential(
            resnet.conv1,    # [B, 64, 112, 112]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # [B, 64, 56, 56]
            resnet.layer1,   # [B, 256, 56, 56]
            resnet.layer2,   # [B, 512, 28, 28]
            resnet.layer3,   # [B, 1024, 14, 14]
            resnet.layer4,   # [B, 2048, 7, 7]
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] input images
        Returns:
            [B, 2048, 7, 7] dense feature map
        """
        return self.features(x)
