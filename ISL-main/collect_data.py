"""
collect_data.py — Preprocess ISL images into 3D MediaPipe landmark + bone
features for BOTH hands. Output: 316-dim vectors (158 per hand).

Usage:
    uv run python collect_data.py --data_dir ./Indian --output ./landmark_data
"""

import os
import pathlib
import numpy as np
from PIL import Image
import argparse
import time
import math
import urllib.request


# ── Bone Connections (20 segments) ───────────────────────────────────
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
]

# Joint triplets for 3D angle computation (parent, joint, child)
JOINT_ANGLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),         # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),         # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),   # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20), # Pinky
]

# 3D dimensions: 63 coords + 60 bone_vecs + 20 bone_lens + 15 angles = 158
SINGLE_HAND_DIM = 158
FEATURE_DIM = SINGLE_HAND_DIM * 2   # 316 — both hands

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), ".cache", "hand_landmarker.task")


# ── 3D Feature Extraction ───────────────────────────────────────────
def compute_single_hand_features_3d(landmarks_3d):
    """
    Compute 158-dim feature vector from 21 (x, y, z) landmarks.

    Layout:
      [0:63]    Normalized 3D coordinates   (21 × 3)
      [63:123]  Bone direction vectors      (20 × 3)
      [123:143] Bone lengths (3D euclidean)  (20)
      [143:158] Inter-joint angles (3D dot)  (15)
    """
    lm = np.array(landmarks_3d, dtype=np.float32)  # (21, 3)

    # Normalize: center on wrist, scale by max 3D distance
    wrist = lm[0].copy()
    lm_c = lm - wrist
    scale = max(np.max(np.linalg.norm(lm_c, axis=1)), 1e-6)
    lm_n = lm_c / scale

    # 1) Normalized 3D coordinates (63 dims)
    coords = lm_n.flatten()

    # 2) 3D Bone direction vectors (60 dims) + 3D bone lengths (20 dims)
    bone_vecs = []
    bone_lens = []
    for a, b in BONES:
        v = lm_n[b] - lm_n[a]                          # (dx, dy, dz)
        bone_vecs.extend(v.tolist())
        bone_lens.append(float(np.linalg.norm(v)))      # sqrt(dx²+dy²+dz²)

    # 3) Inter-joint angles using 3D dot product (15 dims)
    #    θ = acos( (v1 · v2) / (‖v1‖ ‖v2‖) )
    angles = []
    for a, b, c in JOINT_ANGLES:
        v1 = lm_n[a] - lm_n[b]   # 3D vector
        v2 = lm_n[c] - lm_n[b]   # 3D vector
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        cos_a = np.clip(dot / norms, -1.0, 1.0)
        angles.append(math.acos(float(cos_a)))

    return np.concatenate([
        coords,                      # 63
        np.array(bone_vecs),         # 60
        np.array(bone_lens),         # 20
        np.array(angles),            # 15
    ]).astype(np.float32)            # Total: 158


def compute_both_hands_features_3d(hands_landmarks_3d):
    """316-dim: hand1 (158) + hand2 (158, zero-padded if missing)."""
    hand1 = compute_single_hand_features_3d(hands_landmarks_3d[0])
    hand2 = (compute_single_hand_features_3d(hands_landmarks_3d[1])
             if len(hands_landmarks_3d) >= 2
             else np.zeros(SINGLE_HAND_DIM, dtype=np.float32))
    return np.concatenate([hand1, hand2])


# ── MediaPipe Detection (dual API) ──────────────────────────────────
def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("[SETUP] Downloading hand_landmarker model …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[SETUP] Saved to {MODEL_PATH}")


def _create_detector():
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=0.3)
        return ("legacy", hands)
    except (AttributeError, ImportError):
        pass

    _ensure_model()
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        running_mode=RunningMode.IMAGE,
    )
    return ("tasks", HandLandmarker.create_from_options(options))


def extract_all_hands_3d(image_rgb_np, api_type, detector):
    """Detect up to 2 hands. Returns list of [(x, y, z)...] per hand."""
    import mediapipe as mp

    if api_type == "legacy":
        result = detector.process(image_rgb_np)
        if not result.multi_hand_landmarks:
            return []
        return [[(lm.x, lm.y, lm.z) for lm in h.landmark]
                for h in result.multi_hand_landmarks[:2]]
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_np)
        result = detector.detect(mp_image)
        if not result.hand_landmarks:
            return []
        # Tasks API: hand_world_landmarks has 3D; fall back to hand_landmarks
        source = (result.hand_world_landmarks
                  if result.hand_world_landmarks else result.hand_landmarks)
        return [[(lm.x, lm.y, lm.z) for lm in h] for h in source[:2]]


# ── Dataset Preprocessing ───────────────────────────────────────────
def preprocess_dataset(data_dir: str, output_dir: str):
    data_path = pathlib.Path(data_dir)
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"[ERROR] Dataset directory '{data_path}' not found.")
        return

    class_dirs = sorted([d for d in data_path.iterdir()
                         if d.is_dir() and not d.name.startswith(".")])
    class_names = [d.name for d in class_dirs]
    print(f"[DATA] Found {len(class_names)} classes: {class_names}")
    np.save(out_path / "class_names.npy", np.array(class_names))

    print("[MEDIAPIPE] Initializing 3D hand detector (max 2 hands) …")
    api_type, detector = _create_detector()
    print(f"[MEDIAPIPE] Using '{api_type}' API | 3D features enabled")
    print(f"[FEATURES] {SINGLE_HAND_DIM} dims/hand × 2 = {FEATURE_DIM} total dims")

    all_features, all_labels = [], []
    skipped, processed = 0, 0
    total_start = time.time()

    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        image_files = [
            f for f in sorted(class_dir.iterdir())
            if f.is_file()
            and not f.name.startswith("._")
            and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ]

        class_ok, class_skip = 0, 0
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                hands_3d = extract_all_hands_3d(img_np, api_type, detector)

                if not hands_3d:
                    class_skip += 1
                    skipped += 1
                    continue

                features = compute_both_hands_features_3d(hands_3d)
                all_features.append(features)
                all_labels.append(class_idx)
                class_ok += 1
                processed += 1
            except Exception:
                class_skip += 1
                skipped += 1

        print(f"  [{class_idx+1:2d}/{len(class_dirs)}] {class_name:>3s}: "
              f"{class_ok} OK, {class_skip} skipped")

    if api_type == "legacy":
        detector.close()

    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    np.save(out_path / "features.npy", features_arr)
    np.save(out_path / "labels.npy", labels_arr)

    elapsed = time.time() - total_start
    print(f"\n[DONE] Preprocessed {processed} images, skipped {skipped}")
    print(f"[DONE] Feature shape: {features_arr.shape} "
          f"({SINGLE_HAND_DIM}×2 = {FEATURE_DIM} dims)")
    print(f"[DONE] Saved to {out_path}/ in {elapsed:.1f}s")


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ISL images → 3D dual-hand landmark+bone features")
    parser.add_argument("--data_dir", default="./Indian", help="Image dataset root")
    parser.add_argument("--output", default="./landmark_data",
                        help="Output dir for .npy files")
    args = parser.parse_args()
    preprocess_dataset(args.data_dir, args.output)
