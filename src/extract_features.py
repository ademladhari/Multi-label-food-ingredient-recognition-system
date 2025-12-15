import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

# -------------------------
# SET YOUR FOLDERS HERE
# -------------------------
DATA_DIR = "../data"            # your images root folder (can have nested class dirs)
OUTPUT_DIR = "../features"      # where to save features (saves under features/<class_name>/)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD MODEL (ResNet50)
# -------------------------
def select_device(request: str = "auto") -> str:
    if request.lower() == "cpu":
        return "cpu"
    if request.lower() == "cuda":
        # user insists on CUDA; try a quick op to verify kernels run
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        try:
            x = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            return "cuda"
        except Exception as e:
            raise RuntimeError(f"CUDA requested but not usable: {e}")
    # auto
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            return "cuda"
        except Exception as e:
            print(f"Warning: CUDA detected but not usable ({e}); falling back to CPU.")
            return "cpu"
    return "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Select device or auto-detect")
args, _unknown = parser.parse_known_args()

device = select_device(args.device)
print(f"Using device: {device}")
model = resnet50(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC
model.to(device)
model.eval()

# -------------------------
# IMAGE TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img).cpu().numpy().reshape(-1)  # (2048,)
    return feat

# -------------------------
# MAIN EXTRACTION LOOP (recursively find class folders)
# -------------------------
records = []

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def iter_image_dirs(root_dir):
    """Yield tuples of (class_name, image_dir) for any leaf directory that contains images.
    class_name is the leaf folder name.
    """
    for curr_root, dirs, files in os.walk(root_dir):
        # if this directory contains image files, treat it as a class folder
        if any(f.lower().endswith(IMAGE_EXTS) for f in files):
            class_name = os.path.basename(curr_root)
            yield class_name, curr_root


seen = set()
for class_name, class_path in iter_image_dirs(DATA_DIR):
    # avoid processing same class_name+path twice
    key = (class_name, class_path)
    if key in seen:
        continue
    seen.add(key)

    print(f"Processing class: {class_name} @ {class_path}")

    out_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    for fname in tqdm(sorted(os.listdir(class_path))):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(class_path, fname)
        try:
            feat = extract_feature(img_path)
        except Exception as e:
            # skip unreadable images
            print(f"Warning: failed {img_path}: {e}")
            continue

        # Save .npy
        feature_path = os.path.join(out_class_dir, fname + ".npy")
        np.save(feature_path, feat)

        # Record metadata
        records.append({
            "class": class_name,
            "image": fname,
            "image_path": img_path,
            "feature_path": feature_path
        })

# Save CSV index
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "features_index.csv"), index=False)

print("Done! Feature extraction complete.")
