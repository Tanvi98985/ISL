# ISL Sign Language Classification

Classifies Indian Sign Language (ISL) gestures: **A–Z, 0–9** (35 classes).  
Three model approaches are provided, from basic to state-of-the-art.

## Models

| #     | Script                 | Approach                                | Backbone            |
| ----- | ---------------------- | --------------------------------------- | ------------------- |
| 1     | `model1_ultralight.py` | Image CNN                               | MobileNetV2         |
| 2     | `model2_highperf.py`   | Image CNN                               | ResNet50            |
| **3** | **`model_best.py`**    | **MediaPipe Landmarks + Bone Geometry** | **Feed-forward NN** |

### ⭐ Recommended: Model 3 (Landmark + Bone)

The best approach uses **MediaPipe** to extract hand landmarks, then computes **117 features** per image:

- 42 landmark coordinates (21 points × x,y)
- 40 bone direction vectors (20 finger segments)
- 20 bone lengths
- 15 joint angles (finger curl/spread)

This makes the model lighting/background invariant and much faster than image-based CNNs.

## Quick Start

```bash
# Step 1: Preprocess dataset (extract landmarks + bones)
uv run python collect_data.py --data_dir ./Indian

# Step 2: Train the model
uv run python model_best.py --mode train

# Step 3: Launch inference WebUI → http://127.0.0.1:7862
uv run python model_best.py --mode ui
```

## Legacy Image-Based Models

```bash
# Model 1 — UltraLight (MobileNetV2)
uv run python model1_ultralight.py --mode train --data_dir ./Indian
uv run python model1_ultralight.py --mode ui   # → http://127.0.0.1:7860

# Model 2 — HighPerformance (ResNet50)
uv run python model2_highperf.py --mode train --data_dir ./Indian
uv run python model2_highperf.py --mode ui     # → http://127.0.0.1:7861
```

## Dataset

Place the ISL dataset in `./Indian/` with subfolders per class:

```
Indian/
  A/  B/  C/  …  Z/
  1/  2/  3/  …  9/
```


# Best Model
# ISL Recognition — 3D Dual-Hand Bone Geometry + Landmark Cutout

This document outlines the architecture of the ISL recognition system in `model_best.py` and `collect_data.py`.

## Core Concept

Uses **MediaPipe** to extract **3D hand geometry** (x, y, z) from images, then trains a lightweight **Feed-Forward NN** on structural features. With the **Landmark Cutout** augmentation, the model learns to predict correctly even when parts of the hand are occluded.

**Accuracy: 99.95%** on the ISL dataset (35 classes: A-Z, 1-9).

---

## 1. 3D Feature Engineering (`collect_data.py`)

### Per-Hand Features (158 dims)

| Feature Group              | Dims    | Math                                                          |
| -------------------------- | ------- | ------------------------------------------------------------- |
| **Normalized 3D Coords**   | 63      | 21 landmarks × (x, y, z), centered on wrist, scale-normalized |
| **Bone Direction Vectors** | 60      | 20 segments × (dx, dy, dz) between connected joints           |
| **Bone Lengths**           | 20      | 3D Euclidean: √(dx² + dy² + dz²)                              |
| **Inter-Joint Angles**     | 15      | 3D dot product: θ = acos((v₁·v₂) / (‖v₁‖‖v₂‖))                |
| **Total**                  | **158** |                                                               |

### Dual-Hand → 316 dims

- Hand 1 → first 158 features
- Hand 2 → next 158 features (zero-padded if only one hand visible)
- Uses `hand_world_landmarks` (true 3D depth) when available via Tasks API

---

## 2. Landmark Cutout Augmentation (`model_best.py`)

### Problem

Standard image cutout doesn't work on 1D feature vectors. Physical hand occlusion (one hand blocking part of the other) causes landmark extraction failures.

### Solution: `AugmentedDataset`

A custom PyTorch `Dataset` that dynamically simulates occlusion during training:

1. With probability **p=0.3**, selects **1-3 random fingers** (thumb, index, middle, ring, pinky)
2. Looks up **all feature indices** belonging to those fingers via precomputed `FINGER_MASKS`:
   - Coordinate indices (3 per landmark)
   - Bone vector indices (3 per segment)
   - Bone length indices (1 per segment)
   - Joint angle indices (1 per joint)
3. **Zeros out** those indices, forcing the network to classify using remaining visible joints
4. Adds **Gaussian noise** (σ=0.02) to remaining features for regularization

### Finger Index Mapping

```python
# Example: masking index finger zeros out these indices in the 158-dim vector:
#   Coords:     [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  (landmarks 5-8 × 3)
#   Bone vecs:  [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]  (bones 4-7 × 3)
#   Bone lens:  [127, 128, 129, 130]                                (bones 4-7)
#   Angles:     [146, 147, 148]                                     (joints 3-5)
```

---

## 3. Model Architecture

```
StaticGestureNet (237K params):
  Linear(316, 512) → BN → ReLU → Dropout(0.3)
  Linear(512, 256) → BN → ReLU → Dropout(0.3)
  Linear(256, 128) → BN → ReLU → Dropout(0.2)
  Linear(128,  64) → BN → ReLU → Dropout(0.15)
  Linear( 64,  35) → [A-Z, 1-9]
```

### Training Config

- **Loss**: CrossEntropy with label smoothing (0.05)
- **Optimizer**: AdamW (lr=1e-3, wd=1e-4)
- **Schedule**: CosineAnnealing → 1e-6 over 60 epochs
- **Early stopping**: patience=15

---

## 4. Live Inference (`--mode live`)

- Draws **color-coded bone skeletons** for both hands (separate palettes)
- Uses **3D features** for prediction, **2D coords** for rendering
- HUD overlay shows prediction, confidence, hand count
# ISL
