"""
BP4D Dataset Loader
Loads face images and AU labels from the BP4D dataset.

BP4D provides:
  - Face images (video frames, typically already cropped/aligned)
  - Binary AU labels for 12 AUs: AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU14, AU15, AU17, AU23, AU24

Expected directory structure:
  bp4d_root/
  ├── images/
  │   ├── SN001/          (subject folders)
  │   │   ├── T1/         (task folders)
  │   │   │   ├── 0001.jpg
  │   │   │   ├── 0002.jpg
  │   │   │   └── ...
  │   │   └── ...
  │   └── ...
  └── AU_labels/
      ├── SN001/
      │   ├── SN001_T1_au_labels.csv   (columns: frame, AU1, AU2, ..., AU24)
      │   └── ...
      └── ...

NOTE: BP4D has different organizational conventions depending on how you
downloaded/preprocessed it. You may need to adjust the path logic below
to match your specific directory layout.
"""

import os
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# The 12 AUs that BP4D annotates (same order as our model config)
BP4D_AUS = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
            "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]


class BP4DDataset(Dataset):
    """
    BP4D dataset for AU detection.

    Since BP4D only provides AU labels (no expression labels),
    expression labels are derived from AU combinations using FACS rules.
    """

    def __init__(self, root_dir, subjects=None, transform=None, derive_expressions=True):
        """
        Args:
            root_dir: path to the BP4D root directory
            subjects: list of subject IDs to include (e.g., ["SN001", "SN002"]).
                     If None, all subjects are loaded.
            transform: torchvision transforms for image preprocessing.
                      If None, a default transform is applied.
            derive_expressions: if True, derive expression labels from AU combinations
        """
        self.root_dir = root_dir
        self.derive_expressions = derive_expressions
        self.transform = transform or self._default_transform()

        # Collect all (image_path, au_label) pairs
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
        """Scan the dataset directory and build the sample list."""
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "AU_labels")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                f"Expected structure: {self.root_dir}/images/<subject>/<task>/<frame>.jpg"
            )

        # Get subject folders
        if subjects is None:
            subjects = sorted(os.listdir(images_dir))

        for subject in subjects:
            subject_img_dir = os.path.join(images_dir, subject)
            if not os.path.isdir(subject_img_dir):
                continue

            # Find label files for this subject
            subject_label_dir = os.path.join(labels_dir, subject)

            # Look for task folders
            for task in sorted(os.listdir(subject_img_dir)):
                task_img_dir = os.path.join(subject_img_dir, task)
                if not os.path.isdir(task_img_dir):
                    continue

                # Try to find the corresponding label file
                label_file = os.path.join(
                    subject_label_dir, f"{subject}_{task}_au_labels.csv"
                )
                if not os.path.exists(label_file):
                    # Try alternative naming conventions
                    alt_patterns = [
                        os.path.join(subject_label_dir, f"{subject}_{task}.csv"),
                        os.path.join(subject_label_dir, f"{task}.csv"),
                    ]
                    label_file = None
                    for alt in alt_patterns:
                        if os.path.exists(alt):
                            label_file = alt
                            break
                    if label_file is None:
                        continue

                # Parse the label file
                frame_labels = self._parse_label_file(label_file)

                # Match images to labels
                for img_name in sorted(os.listdir(task_img_dir)):
                    if not img_name.lower().endswith((".jpg", ".png", ".bmp")):
                        continue

                    frame_id = os.path.splitext(img_name)[0]
                    # Try both with and without leading zeros
                    frame_key = frame_id.lstrip("0") or "0"

                    if frame_key in frame_labels:
                        img_path = os.path.join(task_img_dir, img_name)
                        au_label = frame_labels[frame_key]
                        self.samples.append((img_path, au_label))

        print(f"BP4D: Loaded {len(self.samples)} samples from {len(subjects)} subjects")

    def _parse_label_file(self, filepath):
        """
        Parse a CSV label file into a dict of {frame_id: au_label_tensor}.
        Expects first row to be header, first column to be frame ID.
        """
        frame_labels = {}
        with open(filepath, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return frame_labels

        # Parse header to find AU column indices
        header = lines[0].strip().split(",")
        au_col_indices = []
        for au_name in BP4D_AUS:
            found = False
            for idx, col in enumerate(header):
                if au_name.lower() in col.strip().lower():
                    au_col_indices.append(idx)
                    found = True
                    break
            if not found:
                # AU not in this file, default to 0
                au_col_indices.append(-1)

        # Parse data rows
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
                    au_values.append(1.0 if val == "1" else 0.0)

            frame_labels[frame_id] = torch.tensor(au_values, dtype=torch.float32)

        return frame_labels

    def _derive_expression(self, au_label):
        """
        Derive an expression label from AU activations using FACS rules.
        Returns an integer class index (0-6).

        Mapping:
          0: Happy    -> AU6 + AU12
          1: Sad      -> AU1 + AU4 + AU15
          2: Fear     -> AU1 + AU2 + AU4
          3: Disgust  -> AU10 + AU17
          4: Anger    -> AU4 + AU7 + AU23
          5: Surprise -> AU1 + AU2
          6: Neutral  -> none active
        """
        # AU indices in our ordering:
        # 0:AU1, 1:AU2, 2:AU4, 3:AU6, 4:AU7, 5:AU10,
        # 6:AU12, 7:AU14, 8:AU15, 9:AU17, 10:AU23, 11:AU24

        scores = torch.zeros(7)

        # Happy: AU6 + AU12
        scores[0] = au_label[3] + au_label[6]
        # Sad: AU1 + AU4 + AU15
        scores[1] = au_label[0] + au_label[2] + au_label[8]
        # Fear: AU1 + AU2 + AU4
        scores[2] = au_label[0] + au_label[1] + au_label[2]
        # Disgust: AU10 + AU17
        scores[3] = au_label[5] + au_label[9]
        # Anger: AU4 + AU7 + AU23
        scores[4] = au_label[2] + au_label[4] + au_label[10]
        # Surprise: AU1 + AU2
        scores[5] = au_label[0] + au_label[1]
        # Neutral: inverse of total AU activation
        scores[6] = max(0, 2.0 - au_label.sum())

        return scores.argmax().item()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, au_label = self.samples[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        result = {
            "image": image,          # [3, 224, 224]
            "au_labels": au_label,   # [12]
        }

        if self.derive_expressions:
            result["expr_label"] = self._derive_expression(au_label)

        return result


def get_bp4d_loaders(root_dir, train_subjects, val_subjects,
                     batch_size=32, num_workers=4):
    """
    Create train and validation dataloaders for BP4D.

    Args:
        root_dir: path to BP4D root
        train_subjects: list of subject IDs for training
        val_subjects: list of subject IDs for validation
        batch_size: batch size
        num_workers: dataloader workers

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
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

    train_ds = BP4DDataset(root_dir, subjects=train_subjects,
                           transform=train_transform)
    val_ds = BP4DDataset(root_dir, subjects=val_subjects,
                         transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
