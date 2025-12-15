import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from model_head import SimpleHead
import numpy as np


def load_ingredient_names(path):
    """Load ingredient names from ingredient_list_label.txt"""
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                names.append(parts[1])
    return names


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(device):
    model = resnet50(weights="IMAGENET1K_V1")
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model


def load_head(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    args = ckpt.get("args", {})
    
    # Infer dimensions
    if "net.0.weight" in state:
        in_dim = state["net.0.weight"].shape[1]
        hidden = state["net.0.weight"].shape[0]
        if "net.3.weight" in state:
            out_dim = state["net.3.weight"].shape[0]
        else:
            out_dim = state["net.2.weight"].shape[0]
    else:
        in_dim = state["net.weight"].shape[1]
        out_dim = state["net.weight"].shape[0]
        hidden = None
    
    dropout = args.get("dropout", 0.0)
    head = SimpleHead(in_features=in_dim, out_features=out_dim, hidden=hidden, dropout=dropout)
    head.load_state_dict(state)
    head.to(device)
    head.eval()
    return head


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(image_path, backbone, head, device):
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = backbone(tensor).cpu().numpy().reshape(-1)
        feat_tensor = torch.from_numpy(feat).unsqueeze(0).to(device)
        logits = head(feat_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    return probs


def main():
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load data
    ingredient_names = load_ingredient_names("data/metadata/ingredient_list_label.txt")
    
    # Load models
    backbone = load_backbone(device)
    head = load_head("checkpoints/allclasses_head_best.pt", device)
    
    # Get all images from test folder
    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print(f"Test folder not found: {test_dir}")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    results = []
    
    for image_file in sorted(image_files):
        image_path = os.path.join(test_dir, image_file)
        print(f"\nProcessing: {image_file}")
        print("-" * 80)
        
        try:
            # Get predictions
            probs = predict(image_path, backbone, head, device)
            
            # Get top 10 predictions
            top_indices = probs.argsort()[-10:][::-1]
            
            # Test different thresholds
            thresholds = [0.50, 0.80, 0.90, 0.95]
            
            result = f"\n{'='*80}\n"
            result += f"IMAGE: {image_file}\n"
            result += f"Path: {image_path}\n"
            result += f"{'-'*80}\n"
            result += f"\nTOP 10 PREDICTIONS:\n"
            
            for rank, idx in enumerate(top_indices, 1):
                name = ingredient_names[idx] if idx < len(ingredient_names) else f"idx_{idx}"
                result += f"  {rank:2d}. {name:<35} prob={probs[idx]:.4f}\n"
            
            result += f"\n{'PREDICTIONS BY THRESHOLD:':<40}\n"
            for threshold in thresholds:
                predictions = [ingredient_names[i] for i in range(len(probs)) 
                              if probs[i] >= threshold and i < len(ingredient_names)]
                predictions = sorted(predictions, key=lambda x: probs[ingredient_names.index(x)], reverse=True)
                
                result += f"\n  Threshold {threshold:.2f} ({len(predictions)} predictions):\n"
                if predictions:
                    for pred in predictions[:15]:  # Show max 15
                        idx = ingredient_names.index(pred)
                        result += f"    • {pred:<33} prob={probs[idx]:.4f}\n"
                    if len(predictions) > 15:
                        result += f"    ... and {len(predictions) - 15} more\n"
                else:
                    result += f"    (No predictions above threshold)\n"
            
            results.append(result)
            print(result)
            
        except Exception as e:
            error_msg = f"\nERROR processing {image_file}: {str(e)}\n"
            results.append(error_msg)
            print(error_msg)
    
    # Save to file
    output_path = "test_folder_results.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TEST FOLDER INFERENCE RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: checkpoints/allclasses_head_best.pt\n")
        f.write(f"Total images tested: {len(image_files)}\n")
        f.write(f"{'='*80}\n")
        f.write(''.join(results))
    
    print(f"\n\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Total images tested: {len(image_files)}")


if __name__ == "__main__":
    main()
