import os
import sys
import random
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from model_head import SimpleHead


def load_class_ingredients(path):
    """Load ground truth ingredient labels for each class."""
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_name = parts[0]
                labels = [int(x) for x in parts[1:]]
                mapping[class_name] = labels
    return mapping


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
    class_ingredients = load_class_ingredients("data/metadata/class_ingredient.txt")
    ingredient_names = load_ingredient_names("data/metadata/ingredient_list_label.txt")
    
    # Load models
    backbone = load_backbone(device)
    head = load_head("checkpoints/allclasses_head_best.pt", device)
    
    # Select random classes to test
    test_classes = ['Greek_salad', 'Fried_chicken', 'Bacon_and_eggs', 'Ham_sandwich', 
                    'Garlic_bread', 'Goulash', 'Hot_and_sour_soup', 'Jambalaya']
    
    results = []
    
    for class_name in test_classes:
        class_dir = f"data/{class_name}"
        if not os.path.exists(class_dir):
            continue
        
        # Get first image from class
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
        
        image_path = os.path.join(class_dir, images[0])
        
        # Get ground truth ingredients
        if class_name not in class_ingredients:
            continue
        
        true_labels = class_ingredients[class_name]
        true_ingredients = [ingredient_names[i] for i, label in enumerate(true_labels) if label == 1 and i < len(ingredient_names)]
        
        # Get predictions
        probs = predict(image_path, backbone, head, device)
        
        # Get top 5 predictions
        top_indices = probs.argsort()[-5:][::-1]
        top_preds = [(ingredient_names[i], probs[i]) for i in top_indices if i < len(ingredient_names)]
        
        # Calculate accuracy metrics with different thresholds
        threshold_low = 0.5
        threshold_high = 0.95
        
        predicted_set_low = set([ingredient_names[i] for i in range(len(probs)) if probs[i] > threshold_low and i < len(ingredient_names)])
        predicted_set_high = set([ingredient_names[i] for i in range(len(probs)) if probs[i] > threshold_high and i < len(ingredient_names)])
        true_set = set(true_ingredients)
        
        correct_low = len(predicted_set_low & true_set)
        correct_high = len(predicted_set_high & true_set)
        total_true = len(true_set)
        total_pred_low = len(predicted_set_low)
        total_pred_high = len(predicted_set_high)
        
        # Format results
        result = f"\n{'='*80}\n"
        result += f"CLASS: {class_name}\n"
        result += f"IMAGE: {images[0]}\n"
        result += f"{'-'*80}\n"
        result += f"GROUND TRUTH INGREDIENTS ({len(true_ingredients)}):\n"
        result += "  " + ", ".join(true_ingredients[:10])
        if len(true_ingredients) > 10:
            result += f"  ... and {len(true_ingredients) - 10} more"
        result += f"\n\nTOP 5 PREDICTIONS (>0.95 threshold):\n"
        for ing, prob in top_preds:
            marker = "✓" if ing in true_set else "✗"
            if prob > threshold_high:
                result += f"  {marker} {ing:30s} prob={prob:.4f}\n"
        
        result += f"\nACCURACY COMPARISON:\n"
        result += f"  Threshold 0.50 (Lenient):\n"
        result += f"    Correct: {correct_low}/{total_true} true ingredients found\n"
        result += f"    Total predicted: {total_pred_low}\n"
        if total_true > 0:
            recall_low = correct_low / total_true * 100
            result += f"    Recall: {recall_low:.1f}%\n"
        
        result += f"\n  Threshold 0.95 (Strict):\n"
        result += f"    Correct: {correct_high}/{total_true} true ingredients found\n"
        result += f"    Total predicted: {total_pred_high}\n"
        if total_true > 0:
            recall_high = correct_high / total_true * 100
            result += f"    Recall: {recall_high:.1f}%\n"
        
        results.append(result)
        print(result)
    
    # Save to file
    output_path = "test_results.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MODEL ACCURACY TEST RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: checkpoints/allclasses_head_best.pt\n")
        f.write(f"Total classes tested: {len(results)}\n")
        f.write(''.join(results))
    
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
