"""
model_best.py — ISL Recognition Model (3D Dual-Hand + Landmark Cutout)

Uses 3D MediaPipe landmarks + bone geometry (158 dims/hand, 316 total).
Implements Landmark Cutout augmentation for occlusion robustness.

Usage:
    uv run python collect_data.py --data_dir ./Indian
    uv run python model_best.py --mode train
    uv run python model_best.py --mode ui
    uv run python model_best.py --mode live
"""

import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from PIL import Image
import gradio as gr
import cv2
import urllib.request as _urllib_request
import mediapipe as _mp


# ── Helpers ──────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Bone & Angle Definitions ────────────────────────────────────────
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

JOINT_ANGLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

# 3D dimensions: 63 + 60 + 20 + 15 = 158 per hand
SINGLE_HAND_DIM = 158
FEATURE_DIM = SINGLE_HAND_DIM * 2  # 316


# ── 3D Feature Extraction ───────────────────────────────────────────
def compute_single_hand_features_3d(landmarks_3d):
    """158-dim feature vector from 21 (x, y, z) landmarks."""
    lm = np.array(landmarks_3d, dtype=np.float32)  # (21, 3)
    wrist = lm[0].copy()
    lm_c = lm - wrist
    scale = max(np.max(np.linalg.norm(lm_c, axis=1)), 1e-6)
    lm_n = lm_c / scale

    coords = lm_n.flatten()  # 63

    bone_vecs, bone_lens = [], []
    for a, b in BONES:
        v = lm_n[b] - lm_n[a]  # 3D vector
        bone_vecs.extend(v.tolist())
        bone_lens.append(float(np.linalg.norm(v)))  # 3D euclidean

    angles = []
    for a, b, c in JOINT_ANGLES:
        v1 = lm_n[a] - lm_n[b]
        v2 = lm_n[c] - lm_n[b]
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angles.append(math.acos(float(np.clip(dot / norms, -1, 1))))

    return np.concatenate([coords, bone_vecs, bone_lens, angles]).astype(np.float32)


def compute_both_hands_features_3d(hands_landmarks_3d):
    """316-dim: hand1 (158) + hand2 (158, zero if missing)."""
    h1 = compute_single_hand_features_3d(hands_landmarks_3d[0])
    h2 = (compute_single_hand_features_3d(hands_landmarks_3d[1])
           if len(hands_landmarks_3d) >= 2
           else np.zeros(SINGLE_HAND_DIM, dtype=np.float32))
    return np.concatenate([h1, h2])


# ── Manual Rule-Based Classification (fallback) ─────────────────────
# Landmark indices: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle,
#                   13-16=ring, 17-20=pinky
# Tip/PIP/MCP per finger:
#   Thumb:  tip=4, IP=3,  MCP=2,  CMC=1
#   Index:  tip=8, DIP=7, PIP=6,  MCP=5
#   Middle: tip=12,DIP=11,PIP=10, MCP=9
#   Ring:   tip=16,DIP=15,PIP=14, MCP=13
#   Pinky:  tip=20,DIP=19,PIP=18, MCP=17

MANUAL_CONFIDENCE = 0.85          # confidence reported for manual matches
NN_FALLBACK_THRESHOLD = 0.60      # use manual rules if NN top-1 conf < this


def _lm_array(landmarks_3d):
    """Convert list of (x,y,z) tuples to (21,3) numpy array, wrist-centred & scaled."""
    lm = np.array(landmarks_3d, dtype=np.float32)
    wrist = lm[0].copy()
    lm_c = lm - wrist
    scale = max(np.max(np.linalg.norm(lm_c, axis=1)), 1e-6)
    return lm_c / scale


def _dist(a, b):
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


def _angle_at(lm, a, b, c):
    """Angle (radians) at vertex b formed by points a-b-c."""
    v1 = lm[a] - lm[b]
    v2 = lm[c] - lm[b]
    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    return math.acos(float(np.clip(dot / norms, -1, 1)))


def _is_finger_extended(lm, finger):
    """
    Return True if finger is extended (straight).
    Uses: tip is far from wrist AND PIP angle is relatively open (> 2.0 rad).
    """
    tip_idx  = {0: 4, 1: 8, 2: 12, 3: 16, 4: 20}[finger]
    pip_idx  = {0: 3, 1: 6, 2: 10, 3: 14, 4: 18}[finger]
    mcp_idx  = {0: 2, 1: 5, 2: 9,  3: 13, 4: 17}[finger]

    if finger == 0:  # thumb — uses different geometry
        # Thumb extended if tip is further from palm centre than IP joint
        palm_center = (lm[0] + lm[5] + lm[17]) / 3.0
        return _dist(lm[4], palm_center) > _dist(lm[3], palm_center) * 1.05

    # For other fingers: tip must be further from wrist than PIP,
    # and angle at PIP should be > ~2.0 rad (relatively straight)
    tip_dist = _dist(lm[tip_idx], lm[0])
    pip_dist = _dist(lm[pip_idx], lm[0])
    pip_angle = _angle_at(lm, mcp_idx, pip_idx, tip_idx - 1)  # DIP = tip_idx - 1
    return tip_dist > pip_dist * 0.85 and pip_angle > 1.8


def _is_finger_curled(lm, finger):
    """Return True if finger is curled (bent into palm)."""
    return not _is_finger_extended(lm, finger)


def _get_extension_state(lm):
    """Return tuple of booleans: (thumb, index, middle, ring, pinky) extended."""
    return tuple(_is_finger_extended(lm, f) for f in range(5))


def manual_classify_landmarks(landmarks_3d):
    """
    Rule-based classification for ISL signs: 1, 2, 5, 8, E, P, X.

    Args:
        landmarks_3d: list of 21 (x, y, z) tuples from MediaPipe

    Returns:
        (label: str, confidence: float) if a rule matches, else None
    """
    lm = _lm_array(landmarks_3d)
    thumb, index, middle, ring, pinky = _get_extension_state(lm)

    # ── Sign "1": only index extended ────────────────────────────────
    if index and not middle and not ring and not pinky and not thumb:
        return ("1", MANUAL_CONFIDENCE)

    # ── Sign "2": index + middle extended (V-shape), others curled ───
    if index and middle and not ring and not pinky and not thumb:
        return ("2", MANUAL_CONFIDENCE)

    # ── Sign "5": all five fingers extended (open palm) ──────────────
    if thumb and index and middle and ring and pinky:
        return ("5", MANUAL_CONFIDENCE)

    # ── Sign "8": thumb, index, middle, ring extended;
    #    middle tip touches thumb tip ──────────────────────────────────
    if index and middle and ring and not pinky:
        mid_thumb_dist = _dist(lm[12], lm[4])  # middle tip → thumb tip
        if mid_thumb_dist < 0.15:  # close enough = touching
            return ("8", MANUAL_CONFIDENCE)

    # ── Sign "E": all fingers curled, thumb across front ─────────────
    if not index and not middle and not ring and not pinky:
        # Thumb should be across the front (near index/middle base)
        thumb_near_index = _dist(lm[4], lm[6]) < 0.25
        thumb_near_middle = _dist(lm[4], lm[10]) < 0.25
        if thumb_near_index or thumb_near_middle:
            return ("E", MANUAL_CONFIDENCE)

    # ── Sign "P": index + middle extended pointing downward,
    #    thumb out to the side ─────────────────────────────────────────
    if index and middle and not ring and not pinky:
        # "P" differs from "2" by fingers pointing DOWN (y of tips > y of MCP)
        # In MediaPipe, y increases downward in image space
        # We use normalised coords where wrist=origin, so check relative positions
        idx_pointing_down = lm[8][1] > lm[5][1]    # index tip below MCP
        mid_pointing_down = lm[12][1] > lm[9][1]   # middle tip below MCP
        if idx_pointing_down and mid_pointing_down and thumb:
            return ("P", MANUAL_CONFIDENCE)

    # ── Sign "X": index finger hooked/bent, others curled ────────────
    if not middle and not ring and not pinky and not thumb:
        # Index must be partially extended but bent at DIP/PIP
        idx_pip_angle = _angle_at(lm, 5, 6, 7)  # MCP→PIP→DIP
        idx_dip_angle = _angle_at(lm, 6, 7, 8)  # PIP→DIP→TIP
        # Hook shape: PIP somewhat open (> 1.2rad) but DIP bent (< 2.2rad)
        idx_tip_dist = _dist(lm[8], lm[0])
        idx_mcp_dist = _dist(lm[5], lm[0])
        if idx_tip_dist > idx_mcp_dist * 0.5 and idx_pip_angle > 1.0 and idx_dip_angle < 2.5:
            return ("X", MANUAL_CONFIDENCE)

    return None


# ── Landmark Cutout Augmentation ─────────────────────────────────────
# Map each finger to its feature indices within a 158-dim single-hand vector.
# Layout: coords[0:63], bone_vecs[63:123], bone_lens[123:143], angles[143:158]

def _build_finger_masks():
    """
    Precompute the feature indices belonging to each of the 5 fingers + wrist.
    Returns dict: finger_name → list of indices in the 158-dim vector.
    """
    finger_landmarks = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }
    finger_bones = {
        "thumb":  [0, 1, 2, 3],
        "index":  [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring":   [12, 13, 14, 15],
        "pinky":  [16, 17, 18, 19],
    }
    finger_angles = {
        "thumb":  [0, 1, 2],
        "index":  [3, 4, 5],
        "middle": [6, 7, 8],
        "ring":   [9, 10, 11],
        "pinky":  [12, 13, 14],
    }

    masks = {}
    for name in finger_landmarks:
        indices = []
        # Coordinate indices (each landmark → 3 values: x, y, z)
        for lm_id in finger_landmarks[name]:
            indices.extend([lm_id * 3, lm_id * 3 + 1, lm_id * 3 + 2])
        # Bone vector indices (each bone → 3 values: dx, dy, dz)
        for bone_id in finger_bones[name]:
            base = 63 + bone_id * 3
            indices.extend([base, base + 1, base + 2])
        # Bone length indices (1 value per bone)
        for bone_id in finger_bones[name]:
            indices.append(123 + bone_id)
        # Angle indices (1 value per joint)
        for angle_id in finger_angles[name]:
            indices.append(143 + angle_id)
        masks[name] = indices
    return masks


FINGER_MASKS = _build_finger_masks()
FINGER_NAMES = list(FINGER_MASKS.keys())


class AugmentedDataset(Dataset):
    """
    Wraps a TensorDataset and applies Landmark Cutout during training.

    With probability `cutout_prob`, randomly selects 1-3 finger clusters
    from one or both hands and zeros out all features belonging to those
    fingers. This simulates physical hand occlusion.

    Optionally adds Gaussian noise to remaining features for extra
    regularization.
    """

    def __init__(self, base_dataset, augment=True, cutout_prob=0.3,
                 max_fingers=3, noise_std=0.02):
        self.base = base_dataset
        self.augment = augment
        self.cutout_prob = cutout_prob
        self.max_fingers = max_fingers
        self.noise_std = noise_std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        if self.augment and random.random() < self.cutout_prob:
            x = x.clone()
            # Pick 1-3 random fingers to mask
            n_fingers = random.randint(1, self.max_fingers)
            fingers = random.sample(FINGER_NAMES, n_fingers)

            for finger in fingers:
                mask_indices = FINGER_MASKS[finger]
                # Apply to hand 1 (indices 0:158)
                if random.random() < 0.7:  # 70% chance to mask hand 1
                    for i in mask_indices:
                        x[i] = 0.0
                # Apply to hand 2 (indices 158:316)
                if random.random() < 0.5:  # 50% chance to mask hand 2
                    for i in mask_indices:
                        if SINGLE_HAND_DIM + i < x.shape[0]:
                            x[SINGLE_HAND_DIM + i] = 0.0

        if self.augment and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x, y


# ── Model ────────────────────────────────────────────────────────────
class StaticGestureNet(nn.Module):
    """Feed-forward NN for 3D dual-hand gesture classification."""

    def __init__(self, input_dim: int = FEATURE_DIM, num_classes: int = 35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ─────────────────────────────────────────────────────────
WEIGHTS_FILE = "best_isl_landmark_model.pth"
DATA_DIR = "./landmark_data"


def train_model(data_dir: str = DATA_DIR):
    device = get_device()
    print(f"[INIT] Device: {device}")

    features_path = os.path.join(data_dir, "features.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    classes_path = os.path.join(data_dir, "class_names.npy")

    if not os.path.exists(features_path):
        print(f"[ERROR] {features_path} not found. Run collect_data.py first.")
        return

    print("[DATA] Loading 3D dual-hand landmark+bone features …")
    X = np.load(features_path)
    y = np.load(labels_path)
    class_names = np.load(classes_path)
    num_classes = len(class_names)
    print(f"[DATA] {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes")
    print(f"[DATA] Feature layout: {SINGLE_HAND_DIM} dims/hand × 2 = {X.shape[1]}")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    base_dataset = TensorDataset(X_t, y_t)
    train_n = int(0.85 * len(base_dataset))
    val_n = len(base_dataset) - train_n
    train_base, val_base = random_split(base_dataset, [train_n, val_n],
                                        generator=torch.Generator().manual_seed(42))
    print(f"[DATA] Split → train={train_n}, val={val_n}")

    # Wrap train set with Landmark Cutout augmentation
    train_ds = AugmentedDataset(train_base, augment=True,
                                cutout_prob=0.3, max_fingers=3, noise_std=0.02)
    val_ds = AugmentedDataset(val_base, augment=False)  # No augmentation for val
    print("[AUGMENT] Landmark Cutout enabled: p=0.3, max_fingers=3, noise=0.02")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, pin_memory=pin)

    print("[MODEL] Building StaticGestureNet (3D dual-hand) …")
    model = StaticGestureNet(input_dim=X.shape[1], num_classes=num_classes).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Input dim: {X.shape[1]} | Params: {total_p:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

    best_val_acc = 0.0
    patience, wait = 15, 0
    epochs = 100
    print(f"[TRAIN] Training up to {epochs} epochs …\n")

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            total += y_batch.size(0)
            correct += pred.eq(y_batch).sum().item()
        train_acc = 100 * correct / total

        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                v_loss += criterion(out, y_batch).item()
                _, pred = out.max(1)
                v_total += y_batch.size(0)
                v_correct += pred.eq(y_batch).sum().item()
        val_acc = 100 * v_correct / v_total
        val_loss = v_loss / len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_val_acc:
            gap = train_acc - val_acc
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train={train_acc:.1f}% val={val_acc:.1f}% gap={gap:.1f}% | "
                  f"loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "input_dim": X.shape[1],
                "class_names": class_names.tolist(),
            }, WEIGHTS_FILE)
            print(f"  ✅ Best model saved ({val_acc:.2f}%)")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break

    print(f"\n[DONE] Best validation accuracy: {best_val_acc:.2f}%")


# ── MediaPipe Detector (cached, dual API) ────────────────────────────
_cached = {}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), ".cache", "hand_landmarker.task")


def _load(device):
    if "model" in _cached:
        return _cached["model"], _cached["class_names"]
    ckpt = torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
    m = StaticGestureNet(input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"])
    m.load_state_dict(ckpt["model_state"])
    m.to(device).eval()
    _cached["model"] = m
    _cached["class_names"] = ckpt["class_names"]
    return m, ckpt["class_names"]


def _get_detector():
    if "detector" in _cached:
        return _cached["api_type"], _cached["detector"]
    try:
        det = _mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=0.3)
        _cached["api_type"], _cached["detector"] = "legacy", det
        return "legacy", det
    except (AttributeError, ImportError):
        pass

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        _urllib_request.urlretrieve(MODEL_URL, MODEL_PATH)
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2, min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3, running_mode=RunningMode.IMAGE)
    det = HandLandmarker.create_from_options(options)
    _cached["api_type"], _cached["detector"] = "tasks", det
    return "tasks", det


def _detect_hands_3d(img_rgb_np, api_type, detector):
    """Detect up to 2 hands. Returns list of [(x,y,z)...] per hand."""
    if api_type == "legacy":
        result = detector.process(img_rgb_np)
        if not result.multi_hand_landmarks:
            return []
        return [[(lm.x, lm.y, lm.z) for lm in h.landmark]
                for h in result.multi_hand_landmarks[:2]]
    else:
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=img_rgb_np)
        result = detector.detect(mp_image)
        if not result.hand_landmarks:
            return []
        source = (result.hand_world_landmarks
                  if result.hand_world_landmarks else result.hand_landmarks)
        return [[(lm.x, lm.y, lm.z) for lm in h] for h in source[:2]]


def _detect_hands_2d(img_rgb_np, api_type, detector):
    """Detect up to 2 hands. Returns list of [(x,y)...] for drawing only."""
    if api_type == "legacy":
        result = detector.process(img_rgb_np)
        if not result.multi_hand_landmarks:
            return []
        return [[(lm.x, lm.y) for lm in h.landmark]
                for h in result.multi_hand_landmarks[:2]]
    else:
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=img_rgb_np)
        result = detector.detect(mp_image)
        if not result.hand_landmarks:
            return []
        return [[(lm.x, lm.y) for lm in h] for h in result.hand_landmarks[:2]]


# ── Inference (Gradio) ───────────────────────────────────────────────
def predict(image):
    if image is None:
        return {"Error — upload an image": 1.0}
    device = get_device()
    try:
        model, class_names = _load(device)
    except FileNotFoundError:
        return {"Error — train the model first (--mode train)": 1.0}

    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

    api_type, detector = _get_detector()
    hands_3d = _detect_hands_3d(img_np, api_type, detector)
    if not hands_3d:
        return {"No hand detected — try a clearer image": 1.0}

    features = compute_both_hands_features_3d(hands_3d)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        top5_p, top5_i = probs.topk(5)
        result = {class_names[top5_i[0][j].item()]: float(top5_p[0][j]) for j in range(5)}

    # Manual fallback: if NN top-1 confidence is low, try rule-based detection
    top1_conf = max(result.values())
    if top1_conf < NN_FALLBACK_THRESHOLD:
        manual = manual_classify_landmarks(hands_3d[0])
        if manual is not None:
            label, m_conf = manual
            # Build result dict with manual prediction on top
            result = {label: m_conf}
            # Add remaining NN predictions (excluding duplicate)
            nn_preds = [
                (class_names[top5_i[0][j].item()], float(top5_p[0][j]))
                for j in range(5)
                if class_names[top5_i[0][j].item()] != label
            ]
            nn_preds.sort(key=lambda kv: -kv[1])
            for k, v in nn_preds[:4]:
                result[k] = v

    return result


def _predict_stream_hold_then_show(image, hold_state: dict, text_state: str):
    """
    Gradio webcam streaming callback:
    - runs prediction every frame
    - only returns the prediction once the same top-1 class is held for ~5 seconds
    - otherwise returns a blank/placeholder output
    """
    HOLD_SECONDS = 2.0
    SHOW_SECONDS = 2.0
    MIN_CONF_FOR_HOLD = 0.70  # 70%+
    COOLDOWN_SECONDS = 0.25

    if hold_state is None:
        hold_state = {}
    hold_state.setdefault("hold_pred", None)
    hold_state.setdefault("hold_started_at", None)
    hold_state.setdefault("last_confirmed_pred", None)
    hold_state.setdefault("last_confirmed_at", 0.0)
    hold_state.setdefault("show_until", 0.0)
    hold_state.setdefault("show_label", None)

    if text_state is None:
        text_state = ""

    now = time.time()

    # If we're currently in the "show confirmed result" window, keep showing it.
    if hold_state["show_label"] is not None and now < float(hold_state["show_until"]):
        return hold_state["show_label"], hold_state, f"Showing result… {(hold_state['show_until'] - now):.1f}s", text_state, text_state
    # Window expired → clear
    if hold_state["show_label"] is not None and now >= float(hold_state["show_until"]):
        hold_state["show_label"] = None
        hold_state["show_until"] = 0.0

    label_dict = predict(image)  # top-5 dict or error/no-hand dict

    # Handle "no hand"/errors: reset hold and don't show predictions
    if (
        not label_dict
        or any(k.startswith("Error") or k.startswith("No hand") for k in label_dict.keys())
    ):
        hold_state["hold_pred"] = None
        hold_state["hold_started_at"] = None
        return {}, hold_state, "Show your hand clearly in the frame…", text_state, text_state

    # Get top-1
    top1_pred = max(label_dict.items(), key=lambda kv: kv[1])[0]
    top1_conf = float(label_dict[top1_pred])  # 0..1

    eligible = top1_conf >= MIN_CONF_FOR_HOLD

    if eligible:
        if top1_pred != hold_state["hold_pred"]:
            hold_state["hold_pred"] = top1_pred
            hold_state["hold_started_at"] = now
        hold_elapsed = now - (hold_state["hold_started_at"] or now)
    else:
        hold_state["hold_pred"] = None
        hold_state["hold_started_at"] = None
        hold_elapsed = 0.0

    confirmed = False
    if eligible and hold_state["hold_pred"] == top1_pred and hold_elapsed >= HOLD_SECONDS:
        if (top1_pred != hold_state["last_confirmed_pred"]) or ((now - hold_state["last_confirmed_at"]) >= COOLDOWN_SECONDS):
            confirmed = True
            hold_state["last_confirmed_pred"] = top1_pred
            hold_state["last_confirmed_at"] = now
            # "Submit" moment for the WebUI:
            print(f"[UI CONFIRMED] {top1_pred} ({top1_conf*100:.1f}%)")

    if confirmed:
        hold_state["show_label"] = label_dict
        hold_state["show_until"] = now + SHOW_SECONDS
        # Reset hold so the user can do the next sign cleanly
        hold_state["hold_pred"] = None
        hold_state["hold_started_at"] = None
        text_state = f"{text_state}{top1_pred}"
        return label_dict, hold_state, f"CONFIRMED: {top1_pred} (showing for {int(SHOW_SECONDS)}s)", text_state, text_state

    if eligible and hold_state["hold_pred"] == top1_pred:
        remaining = max(0.0, HOLD_SECONDS - hold_elapsed)
        return {}, hold_state, f"Hold steady… {remaining:.1f}s (conf {top1_conf*100:.1f}%)", text_state, text_state

    return {}, hold_state, f"Show sign steadily (need ≥ {int(MIN_CONF_FOR_HOLD*100)}% confidence)", text_state, text_state


# ── Gradio UI ────────────────────────────────────────────────────────
def launch_ui():
    # Keep the same overall UX (webcam + upload + prediction on the right),
    # but make webcam truly live (no capture/submit) with a 5s hold-confirm.
    with gr.Blocks(title="🤟 ISL Recognition — 3D Dual-Hand + Bone Model") as demo:
        gr.Markdown(
            "## 🤟 ISL Recognition — 3D Dual-Hand + Bone Model\n"
            "Uses 3D MediaPipe landmarks + bone geometry + landmark cutout for occlusion-robust recognition."
        )

        with gr.Tabs():
            with gr.Tab("Live Webcam"):
                with gr.Row():
                    cam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="pil",
                        label="Live Webcam",
                    )
                    out_live = gr.Label(num_top_classes=5, label="Prediction")
                status = gr.Textbox(label="Status", interactive=False)
                hold_state = gr.State({})
                text_state = gr.State("")
                detected_text = gr.Textbox(label="Detected Text", interactive=False)
                clear_btn = gr.Button("Clear")

                def _clear_text():
                    return "", ""

                clear_btn.click(fn=_clear_text, inputs=None, outputs=[detected_text, text_state])
                cam.stream(
                    fn=_predict_stream_hold_then_show,
                    inputs=[cam, hold_state, text_state],
                    outputs=[out_live, hold_state, status, detected_text, text_state],
                )

            with gr.Tab("Upload Image"):
                with gr.Row():
                    up = gr.Image(type="pil", label="Upload ISL Sign Image")
                    out_up = gr.Label(num_top_classes=5, label="Prediction")
                up.change(fn=predict, inputs=up, outputs=out_up)

    demo.launch(server_name="127.0.0.1", server_port=7865)


# ── Live OpenCV Webcam ───────────────────────────────────────────────
HAND_COLORS = [
    [(0, 255, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)],
    [(255, 255, 0), (255, 165, 0), (0, 255, 128), (255, 128, 0), (200, 0, 200)],
]
BONE_FINGER = [0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4
PALM_CONNECTIONS = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]


def _draw_hand(frame, landmarks_px, hand_idx):
    colors = HAND_COLORS[hand_idx % 2]
    for i, (a, b) in enumerate(BONES):
        cv2.line(frame, landmarks_px[a], landmarks_px[b],
                 colors[BONE_FINGER[i]], 3, cv2.LINE_AA)
    palm_c = (200, 200, 200) if hand_idx == 0 else (180, 180, 255)
    for a, b in PALM_CONNECTIONS:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], palm_c, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(landmarks_px):
        r = 7 if i == 0 else (5 if i in [4, 8, 12, 16, 20] else 4)
        c = (255, 255, 255) if i == 0 else colors[min(4, max(0, (i-1)//4))]
        cv2.circle(frame, (x, y), r, c, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)
    tips = [4, 8, 12, 16, 20]
    names = ["Thm", "Idx", "Mid", "Rng", "Pnk"]
    for idx, tip in enumerate(tips):
        tx, ty = landmarks_px[tip]
        cv2.putText(frame, names[idx], (tx+8, ty-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[idx], 1, cv2.LINE_AA)


def launch_live():
    device = get_device()
    print(f"[LIVE] Device: {device}")
    try:
        model, class_names = _load(device)
        print(f"[LIVE] Model loaded — {len(class_names)} classes")
    except FileNotFoundError:
        print("[ERROR] No trained model. Run --mode train first.")
        return

    api_type, detector = _get_detector()
    print(f"[LIVE] MediaPipe '{api_type}' API ready (2 hands, 3D)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[LIVE] Webcam started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands — 3D for prediction, 2D for drawing
        hands_3d = _detect_hands_3d(rgb, api_type, detector)
        hands_2d = _detect_hands_2d(rgb, api_type, detector)

        if hands_3d:
            # Draw each hand skeleton (2D pixel coords)
            for hi, lm_2d in enumerate(hands_2d):
                px = [(int(x * w), int(y * h)) for x, y in lm_2d]
                _draw_hand(frame, px, hi)

            # Predict using 3D features
            features = compute_both_hands_features_3d(hands_3d)
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)
                top_p, top_i = probs.max(1)
                pred = class_names[top_i.item()]
                conf = top_p.item() * 100

            # Manual fallback for OpenCV live mode
            if conf < NN_FALLBACK_THRESHOLD * 100:
                manual = manual_classify_landmarks(hands_3d[0])
                if manual is not None:
                    pred, m_conf = manual
                    conf = m_conf * 100

            # HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (420, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"Prediction: {pred}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {conf:.1f}%",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Hands: {len(hands_3d)} | 3D+Bones",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "No hand detected", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("ISL Live — 3D Dual-Hand Bone Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[LIVE] Stopped.")


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ISL Model — 3D Dual-Hand + Landmark Cutout")
    p.add_argument("--mode", choices=["train", "ui", "live"], required=True)
    p.add_argument("--data_dir", default=DATA_DIR, help="Preprocessed data dir")
    args = p.parse_args()

    if args.mode == "train":
        train_model(args.data_dir)
    elif args.mode == "ui":
        launch_ui()
    elif args.mode == "live":
        launch_live()