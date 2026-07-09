"""
DISFA Dataset Loader for MuscleAU-Net

Loads face images and AU intensity labels from the DISFA dataset.
Adapted for the new architecture (256x256 input, 12 DISFA AUs).

DISFA AUs: AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26

Expected structure:
  disfa_root/
  ├── img/
  │   ├── SN001/
  │   │   ├── 0.png   (0-indexed)
  │   │   └── ...
  │   └── ...
  └── ActionUnit_Labels/
      ├── SN001/
      │   ├── SN001_au1.txt
      │   └── ...
      └── ...

Labels are 1-indexed (frame 1,2,...), images are 0-indexed (0.png, 1.png,...).
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
DISFA_AU_NAMES = [f"AU{i}" for i in DISFA_AUS]


class DISFADataset(Dataset):
    """DISFA dataset for AU detection."""

    def __init__(self, root_dir, subjects=None, transform=None,
                 intensity_threshold=2):
        self.root_dir = root_dir
        self.intensity_threshold = intensity_threshold
        self.transform = transform or self._default_transform()
        self.num_aus = len(DISFA_AUS)

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
        # Find images directory
        for img_folder in ["img", "Images", "images"]:
            images_dir = os.path.join(self.root_dir, img_folder)
            if os.path.exists(images_dir):
                break

        labels_dir = os.path.join(self.root_dir, "ActionUnit_Labels")
        if not os.path.exists(labels_dir):
            labels_dir = os.path.join(self.root_dir, "actionunit_labels")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found under {self.root_dir}")

        if subjects is None:
            subjects = sorted([
                d for d in os.listdir(images_dir)
                if os.path.isdir(os.path.join(images_dir, d))
            ])

        for subject in subjects:
            subject_img_dir = os.path.join(images_dir, subject)
            subject_label_dir = os.path.join(labels_dir, subject)

            if not os.path.isdir(subject_img_dir):
                continue
            if not os.path.isdir(subject_label_dir):
                continue

            # Load AU labels
            frame_aus = self._load_subject_labels(subject, subject_label_dir)

            # Match images
            img_files = sorted([
                f for f in os.listdir(subject_img_dir)
                if f.lower().endswith((".jpg", ".png", ".bmp"))
            ])

            for img_name in img_files:
                frame_num = self._extract_frame_number(img_name)
                if frame_num is None:
                    continue

                # Images 0-indexed, labels 1-indexed
                label_frame = frame_num + 1
                if label_frame in frame_aus:
                    img_path = os.path.join(subject_img_dir, img_name)
                    au_label = self._binarize(frame_aus[label_frame])
                    self.samples.append((img_path, au_label))

        print(f"DISFA: Loaded {len(self.samples)} samples from {len(subjects)} subjects")

    def _load_subject_labels(self, subject, label_dir):
        frame_aus = {}
        for au_idx, au_num in enumerate(DISFA_AUS):
            candidates = [
                os.path.join(label_dir, f"{subject}_au{au_num}.txt"),
                os.path.join(label_dir, f"au{au_num}.txt"),
            ]
            label_file = None
            for c in candidates:
                if os.path.exists(c):
                    label_file = c
                    break
            if label_file is None:
                continue

            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    frame_num = int(parts[0].strip())
                    intensity = int(parts[1].strip())
                    if frame_num not in frame_aus:
                        frame_aus[frame_num] = [0] * len(DISFA_AUS)
                    frame_aus[frame_num][au_idx] = intensity

        result = {}
        for frame_num, intensities in frame_aus.items():
            result[frame_num] = torch.tensor(intensities, dtype=torch.float32)
        return result

    def _binarize(self, intensities):
        return (intensities >= self.intensity_threshold).float()

    def _extract_frame_number(self, filename):
        name = os.path.splitext(filename)[0]
        parts = name.replace("-", "_").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        if name.isdigit():
            return int(name)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, au_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "au_labels": au_label}


def get_disfa_loaders(root_dir, train_subjects, val_subjects,
                      batch_size=8, num_workers=0, intensity_threshold=2):
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

    train_ds = DISFADataset(root_dir, subjects=train_subjects,
                            transform=train_transform,
                            intensity_threshold=intensity_threshold)
    val_ds = DISFADataset(root_dir, subjects=val_subjects,
                          transform=val_transform,
                          intensity_threshold=intensity_threshold)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader
