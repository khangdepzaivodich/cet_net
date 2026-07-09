"""
ROI (Region of Interest) Definitions for Local Texture Branch

Defines 7 facial ROIs using MediaPipe Face Mesh landmark indices.
Each ROI is defined by a set of landmark indices that form its boundary.
The bounding box is computed from these landmarks at runtime.

MediaPipe Face Mesh provides 468 landmarks.
Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

# ============================================================================
# 7 ROI Regions
# Each ROI is defined by the landmark indices that bound the region.
# At runtime, we compute a bounding box from these landmarks, then crop
# the face image at that region.
# ============================================================================

ROI_DEFINITIONS = {
    "left_eye": {
        "landmarks": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "description": "Left eye region including orbital area",
    },
    "right_eye": {
        "landmarks": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "description": "Right eye region including orbital area",
    },
    "forehead": {
        # MediaPipe doesn't have true forehead landmarks, so we use
        # upper face landmarks and extend upward
        "landmarks": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                      109, 67, 103, 54, 21, 162, 127, 234, 93, 132],
        "description": "Forehead region (approximated from upper face landmarks)",
    },
    "nose": {
        "landmarks": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
                      98, 97, 326, 327, 278, 279, 48, 49, 219, 218],
        "description": "Nose bridge and tip region",
    },
    "mouth": {
        "landmarks": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                      78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                      95, 88, 178, 87, 14, 317, 402, 318, 324],
        "description": "Mouth region including lips",
    },
    "left_cheek": {
        "landmarks": [116, 123, 147, 187, 207, 206, 205, 36, 142, 126,
                      217, 174, 196, 197, 419, 248],
        "description": "Left cheek region (nasolabial fold area)",
    },
    "right_cheek": {
        "landmarks": [345, 352, 376, 411, 427, 426, 425, 266, 371, 355,
                      437, 399, 419, 197, 196, 174],
        "description": "Right cheek region (nasolabial fold area)",
    },
}

ROI_NAMES = list(ROI_DEFINITIONS.keys())
NUM_ROIS = len(ROI_NAMES)  # 7


def get_roi_bbox_from_landmarks(landmarks, roi_name, image_h, image_w, padding=0.15):
    """
    Compute a bounding box for a ROI from MediaPipe landmarks.

    Args:
        landmarks: numpy array of shape [468, 3] (x, y, z normalized to [0,1])
        roi_name: one of ROI_NAMES
        image_h: original image height
        image_w: original image width
        padding: fractional padding to add around the bbox

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds
    """
    roi_lm_indices = ROI_DEFINITIONS[roi_name]["landmarks"]
    roi_points = landmarks[roi_lm_indices, :2]  # [N, 2] (x, y)

    # Get bounding box in normalized coordinates
    x_min, y_min = roi_points.min(axis=0)
    x_max, y_max = roi_points.max(axis=0)

    # Add padding
    w = x_max - x_min
    h = y_max - y_min
    x_min -= w * padding
    y_min -= h * padding
    x_max += w * padding
    y_max += h * padding

    # Convert to pixel coordinates and clamp
    x1 = max(0, int(x_min * image_w))
    y1 = max(0, int(y_min * image_h))
    x2 = min(image_w, int(x_max * image_w))
    y2 = min(image_h, int(y_max * image_h))

    # Ensure minimum size (at least 8x8)
    if x2 - x1 < 8:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - 4)
        x2 = min(image_w, cx + 4)
    if y2 - y1 < 8:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - 4)
        y2 = min(image_h, cy + 4)

    return x1, y1, x2, y2
