"""
DISFA Dataset Loader
Loads face images and AU intensity labels from the DISFA dataset.

DISFA provides:
  - Face images (video frames)
  - AU intensity labels (0-5 scale) for 12 AUs:
    AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26

IMPORTANT: DISFA uses a DIFFERENT set of AUs than BP4D.
  - AUs in DISFA but NOT in our model: AU5, AU9, AU20, AU25, AU26
  - AUs in our model but NOT in DISFA: AU7, AU10, AU14, AU23, AU24
  This loader maps DISFA's AUs to our model's 12-AU ordering,
  setting missing AUs to 0.

Expected directory structure:
  disfa_root/
  ├── img/
  │   ├── SN001/              (subject folders)
  │   │   ├── 0.png           (0-indexed frame images)
  │   │   ├── 1.png
  │   │   └── ...
  │   └── ...
  └── ActionUnit_Labels/
      ├── SN001/
      │   ├── SN001_au1.txt   (one file per AU per subject)
      │   ├── SN001_au2.txt
      │   └── ...
      └── ...

Each AU label file has lines like (1-indexed frames):
  frame_number,intensity_level
  1,0
  2,1
  ...

NOTE: Images are 0-indexed (0.png, 1.png, ...) while labels are
1-indexed (frame 1, 2, ...). The loader handles this offset.
"""

import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# DISFA's 12 AUs
DISFA_AUS = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
             "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]

# Our model's 12 AUs (from config.py)
MODEL_AUS = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
             "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]

# Mapping from our model's AU index to DISFA's AU index
# -1 means this AU is not available in DISFA (will be set to 0)
MODEL_TO_DISFA = []
for au in MODEL_AUS:
    if au in DISFA_AUS:
        MODEL_TO_DISFA.append(DISFA_AUS.index(au))
    else:
        MODEL_TO_DISFA.append(-1)
# Result: [0, 1, 2, 4, -1, -1, 6, -1, 7, 8, -1, -1]
# Meaning: AU1->0, AU2->1, AU4->2, AU6->4, AU7->missing, AU10->missing, etc.


class DISFADataset(Dataset):
    """
    DISFA dataset for AU detection.

    AU intensities (0-5) are binarized: intensity >= threshold -> 1, else -> 0.
    Default threshold is 2 (following standard DISFA evaluation protocol).
    """

    def __init__(self, root_dir, subjects=None, transform=None,
                 intensity_threshold=2, derive_expressions=True):
        """
        Args:
            root_dir: path to the DISFA root directory
            subjects: list of subject IDs to include. If None, all are loaded.
            transform: image transforms. If None, default is applied.
            intensity_threshold: AU intensity >= this is considered "active" (binary 1).
                                Standard protocol uses threshold=2.
            derive_expressions: if True, derive expression labels from AU combinations
        """
        self.root_dir = root_dir
        self.intensity_threshold = intensity_threshold
        self.derive_expressions = derive_expressions
        self.transform = transform or self._default_transform()

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
        # Try common folder names for images
        for img_folder in ["img", "Images", "images"]:
            images_dir = os.path.join(self.root_dir, img_folder)
            if os.path.exists(images_dir):
                break

        labels_dir = os.path.join(self.root_dir, "ActionUnit_Labels")
        if not os.path.exists(labels_dir):
            labels_dir = os.path.join(self.root_dir, "actionunit_labels")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                f"Expected structure: {self.root_dir}/Images/<subject>/"
            )

        # Get subject folders
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
                print(f"  Warning: No labels for subject {subject}, skipping")
                continue

            # Load all AU label files for this subject
            # Result: {frame_number: [12 DISFA AU intensities]}
            frame_aus = self._load_subject_labels(subject, subject_label_dir)

            # Match images to labels
            img_files = sorted([
                f for f in os.listdir(subject_img_dir)
                if f.lower().endswith((".jpg", ".png", ".bmp"))
            ])

            for img_name in img_files:
                # Extract frame number from filename
                frame_num = self._extract_frame_number(img_name)
                if frame_num is None:
                    continue

                # Images are 0-indexed (0.png, 1.png, ...)
                # Labels are 1-indexed (frame 1, 2, ...)
                # So image "0.png" corresponds to label frame 1
                label_frame_num = frame_num + 1

                if label_frame_num in frame_aus:
                    img_path = os.path.join(subject_img_dir, img_name)
                    disfa_intensities = frame_aus[label_frame_num]

                    # Map DISFA's 12 AUs to our model's 12 AUs
                    model_au_label = self._map_to_model_aus(disfa_intensities)

                    self.samples.append((img_path, model_au_label))

        print(f"DISFA: Loaded {len(self.samples)} samples from {len(subjects)} subjects")

    def _load_subject_labels(self, subject, label_dir):
        """
        Load all AU label files for one subject.

        Returns:
            {frame_number: tensor of 12 DISFA AU intensities}
        """
        frame_aus = {}

        for au_idx, au_name in enumerate(DISFA_AUS):
            # Try common naming patterns
            candidates = [
                os.path.join(label_dir, f"{subject}_{au_name.lower()}.txt"),
                os.path.join(label_dir, f"{subject}_au{au_name[2:]}.txt"),
                os.path.join(label_dir, f"{au_name.lower()}.txt"),
                os.path.join(label_dir, f"au{au_name[2:]}.txt"),
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
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue

                    frame_num = int(parts[0].strip())
                    intensity = int(parts[1].strip())

                    if frame_num not in frame_aus:
                        frame_aus[frame_num] = [0] * len(DISFA_AUS)
                    frame_aus[frame_num][au_idx] = intensity

        # Convert to tensors
        result = {}
        for frame_num, intensities in frame_aus.items():
            result[frame_num] = torch.tensor(intensities, dtype=torch.float32)

        return result

    def _map_to_model_aus(self, disfa_intensities):
        """
        Map DISFA's 12 AU intensities to our model's 12-AU binary labels.
        Uses MODEL_TO_DISFA mapping. Missing AUs are set to 0.

        Args:
            disfa_intensities: tensor of [12] DISFA AU intensities (0-5 scale)

        Returns:
            model_au_label: tensor of [12] binary AU labels for our model
        """
        model_label = torch.zeros(len(MODEL_AUS), dtype=torch.float32)

        for model_idx, disfa_idx in enumerate(MODEL_TO_DISFA):
            if disfa_idx == -1:
                # This AU is not in DISFA, leave as 0
                model_label[model_idx] = 0.0
            else:
                # Binarize: intensity >= threshold -> 1
                intensity = disfa_intensities[disfa_idx].item()
                model_label[model_idx] = 1.0 if intensity >= self.intensity_threshold else 0.0

        return model_label

    def _extract_frame_number(self, filename):
        """Extract frame number from a filename like 'SN001_0042.jpg' or '0042.jpg'."""
        name = os.path.splitext(filename)[0]
        # Try to get the last numeric part
        parts = name.replace("-", "_").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        # If the whole name is a number
        if name.isdigit():
            return int(name)
        return None

    def _derive_expression(self, au_label):
        """
        Derive an expression label from AU activations using FACS rules.
        Same logic as BP4D loader.
        """
        scores = torch.zeros(7)
        scores[0] = au_label[3] + au_label[6]                         # Happy: AU6 + AU12
        scores[1] = au_label[0] + au_label[2] + au_label[8]           # Sad: AU1 + AU4 + AU15
        scores[2] = au_label[0] + au_label[1] + au_label[2]           # Fear: AU1 + AU2 + AU4
        scores[3] = au_label[5] + au_label[9]                         # Disgust: AU10 + AU17
        scores[4] = au_label[2] + au_label[4] + au_label[10]          # Anger: AU4 + AU7 + AU23
        scores[5] = au_label[0] + au_label[1]                         # Surprise: AU1 + AU2
        scores[6] = max(0, 2.0 - au_label.sum())                      # Neutral
        return scores.argmax().item()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, au_label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        result = {
            "image": image,          # [3, 224, 224]
            "au_labels": au_label,   # [12] (mapped to model's AU ordering)
        }

        if self.derive_expressions:
            result["expr_label"] = self._derive_expression(au_label)

        return result


def get_disfa_loaders(root_dir, train_subjects, val_subjects,
                      batch_size=32, num_workers=4, intensity_threshold=2):
    """
    Create train and validation dataloaders for DISFA.

    Args:
        root_dir: path to DISFA root
        train_subjects: list of subject IDs for training
        val_subjects: list of subject IDs for validation
        batch_size: batch size
        num_workers: dataloader workers
        intensity_threshold: binarization threshold (default 2)

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

    train_ds = DISFADataset(
        root_dir, subjects=train_subjects, transform=train_transform,
        intensity_threshold=intensity_threshold,
    )
    val_ds = DISFADataset(
        root_dir, subjects=val_subjects, transform=val_transform,
        intensity_threshold=intensity_threshold,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
