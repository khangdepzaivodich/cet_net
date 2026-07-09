"""
MuscleAU-Net Training Script

Full training loop with:
- All 5 loss components (AU, logic, counterfactual, graph, attention)
- Adam optimizer with differential LR (backbone vs rest)
- Cosine annealing scheduler
- Per-batch progress reporting
"""

import os
import sys
import time
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from models.muscleaunet import MuscleAUNet
from losses.au_loss import AULoss
from losses.logic_loss import LogicLoss
from losses.graph_loss import GraphLoss
from losses.counterfactual_loss import CounterfactualLoss
from losses.attention_loss import AttentionLoss


def create_model(cfg, device):
    model = MuscleAUNet(
        backbone_name=cfg.backbone,
        backbone_dim=cfg.backbone_dim,
        hidden_dim=cfg.hidden_dim,
        texture_dim=cfg.texture_dim,
        num_muscles=cfg.num_muscles,
        num_aus=cfg.num_aus,
        au_list=cfg.au_indices,
        num_geometry_predicates=cfg.num_geometry_predicates,
        num_texture_predicates=cfg.num_texture_predicates,
        gat_heads=cfg.gat_heads,
        gat_layers=cfg.gat_layers,
        cross_attn_heads=cfg.cross_attn_heads,
        cross_attn_layers=cfg.cross_attn_layers,
    )
    return model.to(device)


def create_optimizer(model, cfg):
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


def train_one_epoch(model, dataloader, losses, optimizer, device, cfg, epoch):
    model.train()
    running = {"total": 0, "au": 0, "logic": 0, "cf": 0, "graph": 0}
    num_batches = 0
    total_batches = len(dataloader)

    au_loss_fn, logic_loss_fn, cf_loss_fn, graph_loss_fn, attn_loss_fn = losses

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)

        optimizer.zero_grad()

        output = model(images)

        # Compute losses
        l_au = au_loss_fn(output["au_preds"], au_labels)
        l_logic = logic_loss_fn(output["rule_satisfactions"])
        l_cf = cf_loss_fn(output["cf_loss"])
        l_graph = graph_loss_fn(output["compatibility_penalties"])
        l_attn = attn_loss_fn(output["gate_info"])

        total = (
            cfg.lambda_au * l_au
            + cfg.lambda_logic * l_logic
            + cfg.lambda_counterfactual * l_cf
            + cfg.lambda_graph * l_graph
            + cfg.lambda_attention * l_attn
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running["total"] += total.item()
        running["au"] += l_au.item()
        running["logic"] += l_logic.item()
        running["cf"] += l_cf.item()
        running["graph"] += l_graph.item()
        num_batches += 1

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(
                f"  Epoch {epoch+1} | Batch {i+1}/{total_batches} | "
                f"Loss: {total.item():.4f} (AU:{l_au.item():.3f} "
                f"Logic:{l_logic.item():.3f} CF:{l_cf.item():.3f} "
                f"Graph:{l_graph.item():.3f})",
                flush=True,
            )

    for key in running:
        running[key] /= max(num_batches, 1)
    return running


@torch.no_grad()
def evaluate(model, dataloader, losses, device, cfg):
    model.eval()
    running = {"total": 0, "au": 0, "logic": 0}
    correct_aus = 0
    total_aus = 0
    num_batches = 0

    au_loss_fn = losses[0]

    for batch in dataloader:
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)

        output = model(images)

        l_au = au_loss_fn(output["au_preds"], au_labels)
        running["au"] += l_au.item()
        running["total"] += l_au.item()

        # AU accuracy (threshold at 0.5)
        au_binary = (output["au_preds"] > 0.5).float()
        correct_aus += (au_binary == au_labels).sum().item()
        total_aus += au_labels.numel()
        num_batches += 1

    for key in running:
        running[key] /= max(num_batches, 1)

    accuracy = correct_aus / max(total_aus, 1)
    return running, accuracy


def train(cfg=None):
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

    # Optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # Loss functions
    losses = (
        AULoss(),
        LogicLoss(),
        CounterfactualLoss(),
        GraphLoss(),
        AttentionLoss(),
    )

    # Dataloaders
    if cfg.dataset == "disfa":
        from datasets.disfa import get_disfa_loaders
        print(f"\nLoading DISFA from: {cfg.disfa_root}")
        train_loader, val_loader = get_disfa_loaders(
            root_dir=cfg.disfa_root,
            train_subjects=cfg.disfa_train_subjects,
            val_subjects=cfg.disfa_val_subjects,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            intensity_threshold=cfg.disfa_intensity_threshold,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # Training loop
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        start = time.time()

        train_losses = train_one_epoch(
            model, train_loader, losses, optimizer, device, cfg, epoch
        )
        val_losses, val_acc = evaluate(model, val_loader, losses, device, cfg)
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{cfg.epochs} ({elapsed:.1f}s) | "
            f"Train: {train_losses['total']:.4f} | "
            f"Val AU Loss: {val_losses['au']:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                os.path.join(cfg.checkpoint_dir, "best_model.pth"),
            )
            print(f"  -> Saved best model (acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MuscleAU-Net")
    default_cfg = Config()
    for key in dir(default_cfg):
        if not key.startswith("__") and not callable(getattr(default_cfg, key)):
            val = getattr(default_cfg, key)
            if type(val) in (int, float, str):
                parser.add_argument(f"--{key}", type=type(val), default=val)

    args = parser.parse_args()
    cfg = Config()
    for key, val in vars(args).items():
        setattr(cfg, key, val)

    train(cfg)
