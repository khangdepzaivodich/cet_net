"""
Logic-Guided Attention Bias Generator

Translates fuzzy rule satisfactions from the first forward pass
into attention bias matrices for the second (feedback) pass.
"""

import torch
import torch.nn as nn

class LogicAttentionBias(nn.Module):
    def __init__(self, num_rules, num_muscles, num_vit_patches=196, num_geo=8, num_tex=7, num_heads=8):
        super().__init__()
        self.num_muscles = num_muscles
        self.num_heads = num_heads
        
        self.num_vit = num_vit_patches
        self.num_geo = num_geo
        self.num_tex = num_tex
        
        # MLPs to map rule scores to attention biases
        self.vit_bias_proj = nn.Sequential(
            nn.Linear(num_rules, 256),
            nn.ReLU(),
            nn.Linear(256, num_muscles * num_vit_patches)
        )
        
        self.geo_bias_proj = nn.Sequential(
            nn.Linear(num_rules, 128),
            nn.ReLU(),
            nn.Linear(128, num_muscles * num_geo)
        )
        
        self.tex_bias_proj = nn.Sequential(
            nn.Linear(num_rules, 128),
            nn.ReLU(),
            nn.Linear(128, num_muscles * num_tex)
        )
        
    def forward(self, rule_satisfactions):
        """
        Args:
            rule_satisfactions: [B, num_rules] scalar scores
            
        Returns:
            dict containing logic bias masks for each modality
            Shapes: [B * num_heads, 18, S] where S is sequence length
        """
        B = rule_satisfactions.shape[0]
        
        # Generate biases
        # [B, 18, S]
        vit_bias = self.vit_bias_proj(rule_satisfactions).view(B, self.num_muscles, self.num_vit)
        geo_bias = self.geo_bias_proj(rule_satisfactions).view(B, self.num_muscles, self.num_geo)
        tex_bias = self.tex_bias_proj(rule_satisfactions).view(B, self.num_muscles, self.num_tex)
        
        # PyTorch MultiheadAttention expects attn_mask to be [B * num_heads, L_target, L_source]
        # So we repeat the bias across all heads
        vit_bias = vit_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * self.num_heads, self.num_muscles, self.num_vit)
        geo_bias = geo_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * self.num_heads, self.num_muscles, self.num_geo)
        tex_bias = tex_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(B * self.num_heads, self.num_muscles, self.num_tex)
        
        return {
            "vit": vit_bias,
            "geo": geo_bias,
            "tex": tex_bias
        }
