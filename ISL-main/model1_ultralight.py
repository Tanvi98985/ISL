"""
Model 1 — UltraLight ISL Classifier
Uses MobileNetV2 (pretrained on ImageNet) as a lightweight backbone
with a custom classification head for 35 ISL classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import pathlib
import os
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

WEIGHTS_FILE = "ultralight_mobilenet_isl.pth"


# ── Model ────────────────────────────────────────────────────────────
class UltraLightISLNet(nn.Module):
    """MobileNetV2 backbone with a compact classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features          # keep conv layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.last_channel        # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Validation wrapper (apply clean transforms) ─────────────────────
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

    img_size = 224  # MobileNetV2 native input
    batch_size = 64
    epochs = 30
    lr = 3e-4

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
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

    train_n = int(0.85 * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    val_ds_clean = ValidationDataset(val_ds, val_tf)
    print(f"[DATA] Split → train={train_n}, val={val_n}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds_clean, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)

    print("[MODEL] Building UltraLightISLNet (MobileNetV2 backbone) …")
    model = UltraLightISLNet(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_params:,} | Trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience, wait = 7, 0
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

        print(f"[Epoch {epoch+1}/{epochs}] train_acc={train_acc:.2f}% | "
              f"val_acc={val_acc:.2f}% | val_loss={val_loss:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

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
    m = UltraLightISLNet(NUM_CLASSES)
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


def _predict_stream_hold_then_show(image, hold_state: dict, text_state: str):
    """
    Live webcam mode (no capture/submit):
    - predict every frame
    - if the same top-1 stays stable for 5s, show the prediction for 5s
    """
    HOLD_SECONDS = 2.0
    SHOW_SECONDS = 2.0
    MIN_CONF_FOR_HOLD = 0.70
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

    # keep showing confirmed result for SHOW_SECONDS
    if hold_state["show_label"] is not None and now < float(hold_state["show_until"]):
        return hold_state["show_label"], hold_state, f"Showing result… {(hold_state['show_until'] - now):.1f}s", text_state, text_state
    if hold_state["show_label"] is not None and now >= float(hold_state["show_until"]):
        hold_state["show_label"] = None
        hold_state["show_until"] = 0.0

    label_dict = predict(image)
    if not label_dict or any(k.startswith("Error") for k in label_dict.keys()):
        hold_state["hold_pred"] = None
        hold_state["hold_started_at"] = None
        return {}, hold_state, "Show your hand clearly in the frame…", text_state, text_state

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
            print(f"[UI CONFIRMED] {top1_pred} ({top1_conf*100:.1f}%)")

    if confirmed:
        hold_state["show_label"] = label_dict
        hold_state["show_until"] = now + SHOW_SECONDS
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
    # Keep the same idea (upload + prediction), but add a true live webcam mode
    # without capture/submit.
    with gr.Blocks(title="ISL Classification — UltraLight (MobileNetV2)") as demo:
        gr.Markdown(
            "## ISL Classification — UltraLight (MobileNetV2)\n"
            "Upload an ISL character image to classify it, or use Live Webcam.\n"
            "Live Webcam: hold the same sign steady for **5s** → prediction shows for **5s**."
        )

        with gr.Tabs():
            with gr.Tab("Live Webcam"):
                with gr.Row():
                    cam = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Live Webcam")
                    out_live = gr.Label(num_top_classes=3, label="Prediction")
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
                    out_up = gr.Label(num_top_classes=3, label="Prediction")
                up.change(fn=predict, inputs=up, outputs=out_up)

    demo.launch(server_name="127.0.0.1", server_port=7860)


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ISL UltraLight Model")
    p.add_argument("--mode", choices=["train", "ui"], required=True)
    p.add_argument("--data_dir", default="./Indian", help="Dataset root")
    args = p.parse_args()

    if args.mode == "train":
        train_model(args.data_dir)
    else:
        launch_ui()