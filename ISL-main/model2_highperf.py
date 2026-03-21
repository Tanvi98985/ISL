"""
Model 2 — High-Performance ISL Classifier
Uses ResNet50 (pretrained on ImageNet) as a powerful backbone
with a deeper classification head for 35 ISL classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
import pathlib
import os
from collections import Counter
import gradio as gr
from PIL import Image
import time


# ── Helpers ──────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


NUM_CLASSES = 35
CLASS_INDEX_TO_CHAR = {}
for i in range(NUM_CLASSES):
    CLASS_INDEX_TO_CHAR[i] = str(i + 1) if i < 9 else chr(ord('A') + i - 9)

WEIGHTS_FILE = "highperf_resnet50_isl.pth"


# ── Model ────────────────────────────────────────────────────────────
class HighPerformanceISLNet(nn.Module):
    """ResNet50 backbone with a multi-layer classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze early layers (conv1 → layer2) to retain generic features
        for name, param in backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Keep everything except the original FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        in_features = 2048

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Validation wrapper ──────────────────────────────────────────────
class ValidationDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        orig = self.subset.indices[idx]
        path, label = self.subset.dataset.samples[orig]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)


# ── Training ─────────────────────────────────────────────────────────
def train_model(data_dir_path: str):
    device = get_device()
    print(f"[INIT] Device selected: {device}")

    data_dir = pathlib.Path(data_dir_path)
    if not data_dir.exists():
        print(f"[ERROR] Dataset directory '{data_dir}' not found.")
        return

    img_size = 224  # ResNet50 native input
    batch_size = 32
    epochs = 40
    lr = 1e-3

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(12),
        transforms.RandomHorizontalFlip(p=0.15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("[DATA] Loading dataset …")
    def _valid(path: str) -> bool:
        """Skip macOS resource forks and non-image junk."""
        return not os.path.basename(path).startswith("._")

    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tf, is_valid_file=_valid)
    num_classes = len(full_ds.classes)
    print(f"[DATA] Found {num_classes} classes | {len(full_ds)} total images")

    train_n = int(0.88 * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"[DATA] Split → train={train_n}, val={val_n}")

    # Weighted sampler for class balancing
    print("[DATA] Computing class-balanced sampler weights …")
    train_targets = [full_ds.targets[train_ds.indices[i]] for i in range(len(train_ds))]
    counts = Counter(train_targets)
    weights = [1.0 / counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    val_ds_clean = ValidationDataset(val_ds, val_tf)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds_clean, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)

    print("[MODEL] Building HighPerformanceISLNet (ResNet50 backbone) …")
    model = HighPerformanceISLNet(num_classes).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_p:,} | Trainable: {trainable_p:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Separate LR for backbone vs classifier
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "classifier" not in n]
    classifier_params = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},   # lower LR for backbone
        {"params": classifier_params, "lr": lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_acc = 0.0
    patience, wait = 10, 0
    print(f"[TRAIN] Starting training for {epochs} epochs …\n")

    for epoch in range(epochs):
        t0 = time.time()

        # ── train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| loss={loss.item():.4f} acc={100*correct/total:.1f}%")

        train_acc = 100 * correct / total

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                v_loss += criterion(out, labels).item()
                _, pred = out.max(1)
                v_total += labels.size(0)
                v_correct += pred.eq(labels).sum().item()
        val_acc = 100 * v_correct / v_total
        val_loss = v_loss / len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0

        gap = train_acc - val_acc
        emoji = "🏆" if val_acc >= 85 else ("🎯" if val_acc >= 70 else "📈")
        print(f"{emoji} [Epoch {epoch+1}/{epochs}] train_acc={train_acc:.2f}% | "
              f"val_acc={val_acc:.2f}% | gap={gap:.1f}% | "
              f"val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_FILE)
            print(f"  ✅ New best model saved ({val_acc:.2f}%)")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break

    print(f"\n[DONE] Training complete. Best val accuracy: {best_val_acc:.2f}%")


# ── Inference ────────────────────────────────────────────────────────
_cached_model = None

def _load_model(device):
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    m = HighPerformanceISLNet(NUM_CLASSES)
    m.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
    m.to(device).eval()
    _cached_model = m
    return m

def predict(image):
    if image is None:
        return {"Error — upload an image first": 1.0}

    device = get_device()
    try:
        model = _load_model(device)
    except FileNotFoundError:
        return {"Error — train the model first (--mode train)": 1.0}

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    tensor = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        top3_p, top3_i = probs.topk(3)
        return {CLASS_INDEX_TO_CHAR[top3_i[0][j].item()]: float(top3_p[0][j])
                for j in range(3)}


# ── Gradio UI ────────────────────────────────────────────────────────
def launch_ui():
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload ISL Sign Image"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="ISL Classification — HighPerformance (ResNet50)",
        description="Upload an ISL character image to classify it. "
                    "Returns the top-3 predicted characters with confidence.",
    )
    demo.launch(server_name="127.0.0.1", server_port=7861)


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ISL HighPerformance Model")
    p.add_argument("--mode", choices=["train", "ui"], required=True)
    p.add_argument("--data_dir", default="./Indian", help="Dataset root")
    args = p.parse_args()

    if args.mode == "train":
        train_model(args.data_dir)
    else:
        launch_ui()