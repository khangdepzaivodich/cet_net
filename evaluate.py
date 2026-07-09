"""
Evaluation Script for MuscleAU-Net

Evaluates a trained model on a given dataset (DISFA or BP4D).
Computes per-AU F1 score and Accuracy, as well as average F1 (F1-mean).
"""

import os
import sys
import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from models.muscleaunet import MuscleAUNet


def evaluate_model(model, dataloader, device, num_aus, au_names):
    model.eval()
    
    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            au_labels = batch["au_labels"]  # [B, num_aus]
            landmarks = batch["landmarks"].to(device)

            output = model(images, landmarks=landmarks)
            au_preds = output["au_preds"].cpu()  # [B, num_aus]

            # Binarize predictions at 0.5 threshold
            au_preds_bin = (au_preds > 0.5).float()

            all_preds.append(au_preds_bin)
            all_labels.append(au_labels)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(dataloader)} batches")

    # Stack all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()    # [N, num_aus]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # [N, num_aus]

    # Compute metrics per AU
    print("\n--- Evaluation Results ---")
    f1_scores = []
    accuracies = []

    for i in range(num_aus):
        preds_i = all_preds[:, i]
        labels_i = all_labels[:, i]

        f1 = f1_score(labels_i, preds_i, zero_division=0) * 100
        acc = accuracy_score(labels_i, preds_i) * 100

        f1_scores.append(f1)
        accuracies.append(acc)

        print(f"{au_names[i]:<10}: F1 = {f1:5.1f}% | Acc = {acc:5.1f}%")

    avg_f1 = np.mean(f1_scores)
    avg_acc = np.mean(accuracies)

    print("-" * 35)
    print(f"{'Average':<10}: F1 = {avg_f1:5.1f}% | Acc = {avg_acc:5.1f}%")

    return avg_f1, avg_acc


def main(args):
    cfg = Config()
    # Override cfg with args if provided
    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("Loading model...")
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
    ).to(device)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint {args.checkpoint} (Epoch {ckpt.get('epoch', '?')})")
    else:
        print("WARNING: No checkpoint provided, evaluating initialized weights!")

    # Dataloaders
    print(f"Loading {cfg.dataset} validation set...")
    if cfg.dataset == "disfa":
        from datasets.disfa import get_disfa_loaders
        _, val_loader = get_disfa_loaders(
            root_dir=cfg.disfa_root,
            train_subjects=cfg.disfa_train_subjects,
            val_subjects=cfg.disfa_val_subjects,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            intensity_threshold=cfg.disfa_intensity_threshold,
        )
    elif cfg.dataset == "bp4d":
        from datasets.bp4d import get_bp4d_loaders
        _, val_loader = get_bp4d_loaders(
            root_dir=cfg.bp4d_root,
            train_subjects=cfg.bp4d_train_subjects,
            val_subjects=cfg.bp4d_val_subjects,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    evaluate_model(model, val_loader, device, cfg.num_aus, cfg.au_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MuscleAU-Net")
    parser.add_argument("--checkpoint", type=str, help="Path to best_model.pth")
    parser.add_argument("--dataset", type=str, choices=["disfa", "bp4d"])
    parser.add_argument("--batch_size", type=int)
    
    args = parser.parse_args()
    main(args)
