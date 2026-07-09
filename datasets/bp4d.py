"""
BP4D Dataset Loader for MuscleAU-Net

Loads face images and AU labels from the BP4D dataset.
Adapted for the new architecture (256x256 input, 12 AUs).

BP4D AUs: AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU14, AU15, AU17, AU23, AU24
"""

import os
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


BP4D_AUS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]


class BP4DDataset(Dataset):
    """BP4D dataset for AU detection."""

    def __init__(self, root_dir, subjects=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform or self._default_transform()
        self.num_aus = len(BP4D_AUS)
        
        self.samples = []
        self._load_samples(subjects)

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_samples(self, subjects):
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "AU_labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: BP4D dataset not found at {self.root_dir}")
            return

        if subjects is None:
            subjects = sorted(os.listdir(images_dir))

        for subject in subjects:
            subject_img_dir = os.path.join(images_dir, subject)
            subject_label_dir = os.path.join(labels_dir, subject)

            if not os.path.isdir(subject_img_dir) or not os.path.isdir(subject_label_dir):
                continue

            for task in sorted(os.listdir(subject_img_dir)):
                task_img_dir = os.path.join(subject_img_dir, task)
                if not os.path.isdir(task_img_dir):
                    continue

                # Try to find the corresponding label file
                label_file = None
                for alt_name in [f"{subject}_{task}_au_labels.csv", f"{subject}_{task}.csv", f"{task}.csv"]:
                    path = os.path.join(subject_label_dir, alt_name)
                    if os.path.exists(path):
                        label_file = path
                        break
                
                if label_file is None:
                    continue

                frame_labels = self._parse_label_file(label_file)

                # Match images to labels
                for img_name in sorted(os.listdir(task_img_dir)):
                    if not img_name.lower().endswith((".jpg", ".png", ".bmp")):
                        continue

                    frame_id = os.path.splitext(img_name)[0].lstrip("0") or "0"

                    if frame_id in frame_labels:
                        img_path = os.path.join(task_img_dir, img_name)
                        self.samples.append((img_path, frame_labels[frame_id]))

        print(f"BP4D: Loaded {len(self.samples)} samples from {len(subjects)} subjects")

    def _parse_label_file(self, filepath):
        frame_labels = {}
        with open(filepath, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return frame_labels

        # Parse header
        header = lines[0].strip().split(",")
        au_col_indices = []
        for au_idx in BP4D_AUS:
            au_name = f"AU{au_idx}"
            found = False
            for idx, col in enumerate(header):
                if au_name.lower() in col.strip().lower():
                    au_col_indices.append(idx)
                    found = True
                    break
            if not found:
                au_col_indices.append(-1)

        # Parse rows
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue

            frame_id = parts[0].strip().lstrip("0") or "0"
            au_values = []
            for col_idx in au_col_indices:
                if col_idx == -1 or col_idx >= len(parts):
                    au_values.append(0.0)
                else:
                    val = parts[col_idx].strip()
                    # 9 means missing annotation in some BP4D subsets, we'll treat it as 0
                    au_values.append(1.0 if val == "1" else 0.0)

            frame_labels[frame_id] = torch.tensor(au_values, dtype=torch.float32)

        return frame_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, au_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # Reverse normalization for MediaPipe
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_unnorm = image_tensor * std + mean
        img_np = (img_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Lazy init MediaPipe FaceMesh (once per worker)
        if not hasattr(self, 'face_mesh'):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True
            )

        results = self.face_mesh.process(img_np)
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face.landmark], dtype=np.float32)
        else:
            landmarks = np.zeros((478, 3), dtype=np.float32) # Fallback if no face detected

        return {"image": image_tensor, "au_labels": au_label, "landmarks": torch.from_numpy(landmarks)}


def get_bp4d_loaders(root_dir, train_subjects, val_subjects,
                     batch_size=8, num_workers=0):
    train_transform = transforms.Compose([
        transforms.Resize((252, 252)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = BP4DDataset(root_dir, subjects=train_subjects, transform=train_transform)
    val_ds = BP4DDataset(root_dir, subjects=val_subjects, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader
