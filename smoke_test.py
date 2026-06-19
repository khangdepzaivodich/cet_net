"""
CET-Net Smoke Test
Verifies that the full model runs without errors and produces correct output shapes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.cet_net import CETNet
from losses import CETNetLoss


def smoke_test():
    print("=" * 60)
    print("CET-Net Smoke Test")
    print("=" * 60)

    device = torch.device("cpu")
    B = 2  # batch size for testing

    # --- 1. Create model ---
    print("\n[1] Creating CET-Net model...")
    model = CETNet(
        num_aus=12,
        num_expressions=7,
        backbone_feat_dim=2048,
        hidden_dim=256,
        spatial_size=7,
        gnn_layers=2,
        pretrained_backbone=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  OK")

    # --- 2. Forward pass ---
    print("\n[2] Running forward pass with random input [2, 3, 224, 224]...")
    x = torch.randn(B, 3, 224, 224, device=device)
    output = model(x)

    print("  Output shapes:")
    for key, val in output.items():
        print(f"    {key:20s}: {list(val.shape)}")

    # Verify shapes
    assert output["expr_probs"].shape == (B, 7), f"expr_probs shape mismatch"
    assert output["expr_logits"].shape == (B, 7), f"expr_logits shape mismatch"
    assert output["beliefs_init"].shape == (B, 12), f"beliefs_init shape mismatch"
    assert output["beliefs_final"].shape == (B, 12), f"beliefs_final shape mismatch"
    assert output["uncertainty"].shape == (B, 12), f"uncertainty shape mismatch"
    assert output["masks"].shape == (B, 12, 7, 7), f"masks shape mismatch"
    print("  All shapes correct!")

    # --- 3. Check beliefs are valid probabilities ---
    print("\n[3] Checking belief values are in [0, 1]...")
    assert output["beliefs_init"].min() >= 0 and output["beliefs_init"].max() <= 1
    assert output["beliefs_final"].min() >= 0 and output["beliefs_final"].max() <= 1
    assert output["uncertainty"].min() >= 0 and output["uncertainty"].max() <= 1
    print("  All values in valid range!")

    # --- 4. Check factor graph changes beliefs ---
    print("\n[4] Checking factor graph modifies beliefs...")
    diff = (output["beliefs_final"] - output["beliefs_init"]).abs().mean().item()
    print(f"  Mean absolute belief change: {diff:.6f}")
    if diff > 1e-6:
        print("  Factor graph IS modifying beliefs (good!)")
    else:
        print("  WARNING: Factor graph is not changing beliefs much")

    # --- 5. Check expression probabilities sum to 1 ---
    print("\n[5] Checking expression probabilities sum to 1...")
    prob_sums = output["expr_probs"].sum(dim=-1)
    print(f"  Probability sums: {prob_sums.tolist()}")
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5)
    print("  Probabilities correctly sum to 1!")

    # --- 6. Test loss computation ---
    print("\n[6] Testing loss computation...")
    criterion = CETNetLoss(
        lambda_au=1.0, lambda_expr=1.0, lambda_rule=0.5, lambda_cf=0.3, num_aus=12
    )
    fake_au_labels = torch.randint(0, 2, (B, 12)).float().to(device)
    fake_expr_labels = torch.randint(0, 7, (B,)).to(device)

    total_loss, loss_dict = criterion(
        output, fake_au_labels, fake_expr_labels, model.expression_head
    )
    print(f"  Total loss:          {loss_dict['total']:.4f}")
    print(f"  AU loss:             {loss_dict['au']:.4f}")
    print(f"  Expression loss:     {loss_dict['expr']:.4f}")
    print(f"  Rule loss:           {loss_dict['rule']:.4f}")
    print(f"  Counterfactual loss: {loss_dict['cf']:.4f}")

    # --- 7. Test backward pass ---
    print("\n[7] Testing backward pass...")
    total_loss.backward()
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    print(f"  {len(grad_norms)} parameters have gradients")
    print(f"  Mean gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")
    print("  Backward pass OK!")

    # --- 8. Print sample output ---
    print("\n[8] Sample output for first image:")
    print(f"  Initial AU beliefs:  {output['beliefs_init'][0].detach().tolist()}")
    print(f"  Final AU beliefs:    {output['beliefs_final'][0].detach().tolist()}")
    print(f"  Uncertainty:         {output['uncertainty'][0].detach().tolist()}")
    print(f"  Expression probs:    {output['expr_probs'][0].detach().tolist()}")
    expr_names = ["Happy", "Sad", "Fear", "Disgust", "Anger", "Surprise", "Neutral"]
    pred_idx = output["expr_probs"][0].argmax().item()
    print(f"  Predicted expression: {expr_names[pred_idx]}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
