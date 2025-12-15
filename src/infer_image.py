import argparse
import os
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from model_head import SimpleHead


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_ingredient_list(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            name = parts[1].strip()
            # ensure list is large enough
            if len(names) <= idx:
                names.extend([""] * (idx - len(names) + 1))
            names[idx] = name
    return names


def infer_head_dims(state_dict: dict) -> Tuple[int, int, int]:
    """Infer (in_dim, out_dim, hidden) from the checkpoint weights."""
    # Hidden MLP case uses net.0.weight (first linear) and last linear layer
    if any(k.startswith("net.0.weight") for k in state_dict.keys()):
        hidden = state_dict["net.0.weight"].shape[0]
        in_dim = state_dict["net.0.weight"].shape[1]
        # Find the last linear layer (could be net.2 or net.3 depending on dropout)
        if "net.3.weight" in state_dict:
            out_dim = state_dict["net.3.weight"].shape[0]
        elif "net.2.weight" in state_dict:
            out_dim = state_dict["net.2.weight"].shape[0]
        else:
            raise KeyError("Cannot find output layer in state_dict")
    else:
        # Single linear layer case
        hidden = None
        in_dim = state_dict["net.weight"].shape[1]
        out_dim = state_dict["net.weight"].shape[0]
    return in_dim, out_dim, hidden


def build_head(checkpoint_path: str, device: torch.device) -> SimpleHead:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    ckpt_args = ckpt.get("args", {})
    dropout = ckpt_args.get("dropout", 0.0)

    in_dim, out_dim, inferred_hidden = infer_head_dims(state)
    hidden = ckpt_args.get("hidden", inferred_hidden)

    head = SimpleHead(in_features=in_dim, out_features=out_dim, hidden=hidden, dropout=dropout)
    head.load_state_dict(state)
    head.to(device)
    head.eval()
    return head


def load_backbone(device: torch.device):
    model = resnet50(weights="IMAGENET1K_V1")
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_inference(image_path: str, backbone, head, device: torch.device):
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = backbone(tensor)
        feats = torch.flatten(feats, 1)  # (1, feat_dim)
        logits = head(feats)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs


def save_probs_csv(save_path: str, image_path: str, probs):
    import csv

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    header = ["image"] + [f"prob_{i}" for i in range(len(probs))]
    with open(save_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow([image_path] + probs.tolist())


def main():
    parser = argparse.ArgumentParser(description="Single-image ingredient probability inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/Aburaage_head.pt", help="Path to trained head checkpoint")
    parser.add_argument("--ingredient_list_path", type=str, default="../data/metadata/ingredient_list_label.txt", help="Path to ingredient index->name mapping")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--top_k", type=int, default=5, help="Show top-K ingredients")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save probabilities as CSV")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    ingredient_names = load_ingredient_list(args.ingredient_list_path)

    backbone = load_backbone(device)
    head = build_head(args.checkpoint_path, device)

    probs = run_inference(args.image_path, backbone, head, device)

    # Basic sanity check on dimension alignment
    if len(ingredient_names) != len(probs):
        print(f"Warning: ingredient list length ({len(ingredient_names)}) != probs length ({len(probs)}). Proceeding, but names may misalign.")

    # Show top-K
    import numpy as np

    top_k = min(args.top_k, len(probs))
    idxs = np.argsort(-probs)[:top_k]
    print("Top predictions:")
    for rank, idx in enumerate(idxs, start=1):
        name = ingredient_names[idx] if idx < len(ingredient_names) else f"idx_{idx}"
        print(f"{rank:2d}. {name:<25} prob={probs[idx]:.4f}")

    # Optional CSV save
    if args.save_csv:
        save_probs_csv(args.save_csv, args.image_path, probs)
        print(f"Saved probabilities to {args.save_csv}")


if __name__ == "__main__":
    main()
