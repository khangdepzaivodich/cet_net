"""
CET-Net Training Script

Standard PyTorch training loop with:
- Adam optimizer with differential learning rates (backbone vs rest)
- Cosine annealing LR scheduler
- All 4 loss components logged per epoch
- Best model checkpoint saved by validation expression accuracy
- Supports BP4D and DISFA datasets (set cfg.dataset to 'bp4d' or 'disfa')
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from models.cet_net import CETNet
from losses import CETNetLoss


def create_model(cfg, device):
    """Create and return the CET-Net model."""
    model = CETNet(
        num_aus=cfg.num_aus,
        num_expressions=cfg.num_expressions,
        backbone_feat_dim=cfg.backbone_feat_dim,
        hidden_dim=cfg.hidden_dim,
        spatial_size=cfg.spatial_size,
        gnn_layers=cfg.gnn_layers,
        pretrained_backbone=True,
    )
    return model.to(device)


def create_optimizer(model, cfg):
    """Create Adam optimizer with differential learning rates."""
    param_groups = [
        {
            "params": model.get_backbone_params(),
            "lr": cfg.lr * cfg.backbone_lr_factor,
            "name": "backbone",
        },
        {
            "params": model.get_non_backbone_params(),
            "lr": cfg.lr,
            "name": "non_backbone",
        },
    ]
    return Adam(param_groups, weight_decay=cfg.weight_decay)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch. Returns average losses."""
    model.train()
    running_losses = {"total": 0, "au": 0, "expr": 0, "rule": 0, "cf": 0}
    num_batches = 0
    total_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)        # [B, 3, 224, 224]
        au_labels = batch["au_labels"].to(device)  # [B, K]
        expr_labels = batch["expr_label"].to(device)  # [B]

        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Compute loss
        loss, loss_dict = criterion(
            output, au_labels, expr_labels, model.expression_head
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate
        for key in running_losses:
            running_losses[key] += loss_dict[key]
        num_batches += 1

        # Print progress every 10 batches
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  Epoch {epoch+1} | Batch {i+1}/{total_batches} | Loss: {loss.item():.4f}", flush=True)

    # Average
    for key in running_losses:
        running_losses[key] /= max(num_batches, 1)

    return running_losses


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set. Returns average losses and accuracy."""
    model.eval()
    running_losses = {"total": 0, "au": 0, "expr": 0, "rule": 0, "cf": 0}
    correct = 0
    total = 0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)
        expr_labels = batch["expr_label"].to(device)

        output = model(images)
        loss, loss_dict = criterion(
            output, au_labels, expr_labels, model.expression_head
        )

        # Expression accuracy
        preds = output["expr_probs"].argmax(dim=-1)
        correct += (preds == expr_labels).sum().item()
        total += expr_labels.size(0)

        for key in running_losses:
            running_losses[key] += loss_dict[key]
        num_batches += 1

    for key in running_losses:
        running_losses[key] /= max(num_batches, 1)

    accuracy = correct / max(total, 1)
    return running_losses, accuracy


def train(cfg=None):
    """Main training function."""
    if cfg is None:
        cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = create_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # Create loss function
    criterion = CETNetLoss(
        lambda_au=cfg.lambda_au,
        lambda_expr=cfg.lambda_expr,
        lambda_rule=cfg.lambda_rule,
        lambda_cf=cfg.lambda_cf,
        num_aus=cfg.num_aus,
    )

    # Create dataloaders based on config
    if cfg.dataset == "bp4d":
        from datasets.bp4d import get_bp4d_loaders
        print(f"\nLoading BP4D dataset from: {cfg.bp4d_root}")
        train_loader, val_loader = get_bp4d_loaders(
            root_dir=cfg.bp4d_root,
            train_subjects=cfg.bp4d_train_subjects,
            val_subjects=cfg.bp4d_val_subjects,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
    elif cfg.dataset == "disfa":
        from datasets.disfa import get_disfa_loaders
        print(f"\nLoading DISFA dataset from: {cfg.disfa_root}")
        train_loader, val_loader = get_disfa_loaders(
            root_dir=cfg.disfa_root,
            train_subjects=cfg.disfa_train_subjects,
            val_subjects=cfg.disfa_val_subjects,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            intensity_threshold=cfg.disfa_intensity_threshold,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}. Use 'bp4d' or 'disfa'.")

    # Training loop
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        start = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_losses, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start

        # Print progress
        print(
            f"Epoch {epoch+1}/{cfg.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_losses['total']:.4f} "
            f"(AU:{train_losses['au']:.3f} Expr:{train_losses['expr']:.3f} "
            f"Rule:{train_losses['rule']:.3f} CF:{train_losses['cf']:.3f}) | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_losses["total"],
                },
                os.path.join(cfg.checkpoint_dir, "best_model.pth"),
            )
            print(f"  -> Saved best model (acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train()
