"""
Stage 2: Geometry Branch

Uses MediaPipe Face Mesh (478 landmarks) to extract facial geometry,
then computes 8 differentiable geometric predicates.

MediaPipe runs on CPU and is NOT differentiable.
The landmark extraction is a preprocessing step.
The predicates computed FROM landmarks use learned parameters and ARE differentiable.

Input:  [B, 3, H, W] raw RGB images (uint8 or float)
Output:
  - geometry_embedding: [B, D_geo] projected geometry features
  - geometry_predicates: [B, 8] predicate truth values in [0, 1]

The 8 predicates:
  0: mouth_corner_up      (corner pulled up → smile)
  1: mouth_corner_down    (corner pulled down → frown)
  2: eyebrow_raised       (brows go up)
  3: brows_together       (brows pulled in)
  4: lip_distance          (lips apart)
  5: eye_openness         (eyes wide vs. squinting)
  6: cheek_raised         (cheek pushed up)
  7: chin_raised          (chin pushed up)
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("WARNING: mediapipe not installed. Geometry branch will use dummy landmarks.")


# Key landmark indices for computing geometric predicates
# (MediaPipe Face Mesh 468 landmarks)
LM = {
    # Mouth corners
    "mouth_left": 61,
    "mouth_right": 291,
    # Upper/lower lip
    "upper_lip": 13,
    "lower_lip": 14,
    # Eyebrow inner points
    "left_brow_inner": 107,
    "right_brow_inner": 336,
    # Eyebrow outer points
    "left_brow_outer": 70,
    "right_brow_outer": 300,
    # Eye upper/lower for openness
    "left_eye_upper": 159,
    "left_eye_lower": 145,
    "right_eye_upper": 386,
    "right_eye_lower": 374,
    # Nose tip and bridge
    "nose_tip": 1,
    "nose_bridge": 6,
    # Cheek (under eye)
    "left_cheek": 123,
    "right_cheek": 352,
    # Chin
    "chin": 152,
    # Reference points for normalization
    "left_ear": 234,
    "right_ear": 454,
    "forehead": 10,
}


class LandmarkExtractor:
    """
    Extracts MediaPipe Face Mesh landmarks from images.
    Runs on CPU. Not differentiable.
    """

    def __init__(self):
        if HAS_MEDIAPIPE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        else:
            self.face_mesh = None

    def extract(self, image_np):
        """
        Extract landmarks from a single image.

        Args:
            image_np: numpy array [H, W, 3] uint8 RGB

        Returns:
            landmarks: [468, 3] numpy array of (x, y, z) normalized to [0,1]
                       or None if no face detected
        """
        if self.face_mesh is None:
            return None

        results = self.face_mesh.process(image_np)
        if results.multi_face_landmarks is None:
            return None

        face = results.multi_face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face.landmark],
            dtype=np.float32,
        )
        return landmarks

    def extract_batch(self, images_np):
        """
        Extract landmarks from a batch of images.

        Args:
            images_np: list of numpy arrays [H, W, 3] uint8

        Returns:
            list of landmark arrays (None for failed detections)
        """
        return [self.extract(img) for img in images_np]


class GeometryPredicateComputer(nn.Module):
    """
    Computes 8 differentiable geometric predicates from raw landmark distances.

    Each predicate is:
      raw_distance → learned_linear → sigmoid → truth_value ∈ [0, 1]

    The learned linear layers allow the network to calibrate what "raised"
    or "open" means in continuous terms, avoiding hardcoded thresholds.
    """

    def __init__(self, num_predicates=8, hidden_dim=32):
        super().__init__()
        self.num_predicates = num_predicates

        # Each predicate: raw_features → hidden → sigmoid → truth value
        # Input: 2 raw distance features per predicate (measurement + reference)
        self.predicate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_predicates)
        ])

    def forward(self, raw_features):
        """
        Args:
            raw_features: [B, 8, 2] — 8 predicates, each with 2 raw measurements

        Returns:
            predicates: [B, 8] truth values in [0, 1]
        """
        preds = []
        for i, net in enumerate(self.predicate_nets):
            preds.append(net(raw_features[:, i, :]))  # [B, 1]
        return torch.cat(preds, dim=-1)  # [B, 8]


class GeometryBranch(nn.Module):
    """
    Full geometry branch: landmarks → raw distances → predicates → embedding.
    """

    def __init__(self, num_predicates=8, embed_dim=256):
        super().__init__()
        self.num_predicates = num_predicates
        self.landmark_extractor = LandmarkExtractor()

        # Predicate computer (differentiable)
        self.predicate_computer = GeometryPredicateComputer(num_predicates)

        # Project predicates to embedding space
        self.projection = nn.Sequential(
            nn.Linear(num_predicates, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def _compute_raw_features(self, landmarks_batch):
        """
        Compute raw distance features from landmarks for each predicate.

        Args:
            landmarks_batch: [B, 468, 3] tensor of landmark coordinates

        Returns:
            raw_features: [B, 8, 2] tensor
        """
        B = landmarks_batch.shape[0]
        device = landmarks_batch.device
        raw = torch.zeros(B, self.num_predicates, 2, device=device)

        lm = landmarks_batch  # [B, 468, 3]

        # Face width for normalization
        face_width = torch.norm(
            lm[:, LM["left_ear"], :2] - lm[:, LM["right_ear"], :2], dim=-1, keepdim=True
        )  # [B, 1]
        face_width = face_width.clamp(min=1e-6)

        # 0: mouth_corner_up — how much mouth corners are above mouth center
        mouth_center_y = (lm[:, LM["upper_lip"], 1] + lm[:, LM["lower_lip"], 1]) / 2
        corner_avg_y = (lm[:, LM["mouth_left"], 1] + lm[:, LM["mouth_right"], 1]) / 2
        raw[:, 0, 0] = (mouth_center_y - corner_avg_y) / face_width.squeeze(-1)
        raw[:, 0, 1] = face_width.squeeze(-1)  # reference

        # 1: mouth_corner_down — inverse of corner_up
        raw[:, 1, 0] = (corner_avg_y - mouth_center_y) / face_width.squeeze(-1)
        raw[:, 1, 1] = face_width.squeeze(-1)

        # 2: eyebrow_raised — distance from brow to eye
        brow_h = (
            lm[:, LM["left_brow_inner"], 1] + lm[:, LM["right_brow_inner"], 1]
        ) / 2
        eye_h = (
            lm[:, LM["left_eye_upper"], 1] + lm[:, LM["right_eye_upper"], 1]
        ) / 2
        raw[:, 2, 0] = (eye_h - brow_h) / face_width.squeeze(-1)
        raw[:, 2, 1] = face_width.squeeze(-1)

        # 3: brows_together — distance between inner brow points
        brow_dist = torch.norm(
            lm[:, LM["left_brow_inner"], :2] - lm[:, LM["right_brow_inner"], :2],
            dim=-1
        )
        raw[:, 3, 0] = brow_dist / face_width.squeeze(-1)
        raw[:, 3, 1] = face_width.squeeze(-1)

        # 4: lip_distance — vertical distance between upper and lower lip
        lip_dist = torch.abs(lm[:, LM["upper_lip"], 1] - lm[:, LM["lower_lip"], 1])
        raw[:, 4, 0] = lip_dist / face_width.squeeze(-1)
        raw[:, 4, 1] = face_width.squeeze(-1)

        # 5: eye_openness — average eye opening (upper - lower lid)
        left_eye_open = torch.abs(
            lm[:, LM["left_eye_upper"], 1] - lm[:, LM["left_eye_lower"], 1]
        )
        right_eye_open = torch.abs(
            lm[:, LM["right_eye_upper"], 1] - lm[:, LM["right_eye_lower"], 1]
        )
        raw[:, 5, 0] = ((left_eye_open + right_eye_open) / 2) / face_width.squeeze(-1)
        raw[:, 5, 1] = face_width.squeeze(-1)

        # 6: cheek_raised — how much cheek is pushed up
        cheek_h = (lm[:, LM["left_cheek"], 1] + lm[:, LM["right_cheek"], 1]) / 2
        nose_h = lm[:, LM["nose_tip"], 1]
        raw[:, 6, 0] = (nose_h - cheek_h) / face_width.squeeze(-1)
        raw[:, 6, 1] = face_width.squeeze(-1)

        # 7: chin_raised — how much chin is pushed up
        chin_h = lm[:, LM["chin"], 1]
        raw[:, 7, 0] = (chin_h - lm[:, LM["lower_lip"], 1]) / face_width.squeeze(-1)
        raw[:, 7, 1] = face_width.squeeze(-1)

        return raw

    def forward(self, images, landmarks=None):
        """
        Args:
            images: [B, 3, H, W] normalized images (for shape info only)
            landmarks: [B, 468, 3] pre-extracted landmarks tensor.
                       If None, uses MediaPipe to extract (slow, CPU).

        Returns:
            geometry_embedding: [B, embed_dim]
            geometry_predicates: [B, 8] truth values in [0, 1]
        """
        if landmarks is None:
            # Fallback: generate random landmarks for shape compatibility
            B = images.shape[0]
            device = images.device
            landmarks = torch.rand(B, 468, 3, device=device)

        # Compute raw distance features
        raw_features = self._compute_raw_features(landmarks)  # [B, 8, 2]

        # Compute differentiable predicates
        predicates = self.predicate_computer(raw_features)  # [B, 8]

        # Project to embedding space
        embedding = self.projection(predicates)  # [B, embed_dim]

        return embedding, predicates
