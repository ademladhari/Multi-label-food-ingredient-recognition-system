import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from multilabel_dataset_all import FeatureMultilabelAllDataset, load_class_ingredient_map
from model_head import SimpleHead


def compute_pos_weight(mapping: dict):
    """Compute per-label pos_weight = (N - pos) / pos for BCEWithLogitsLoss.
    mapping: class_name -> binary vector
    """
    # Stack all class vectors to get counts per ingredient across classes
    arr = np.stack(list(mapping.values()), axis=0)  # (num_classes, num_labels)
    pos = arr.sum(axis=0)
    N = arr.shape[0]
    # avoid division by zero
    pos = np.clip(pos, 1.0, None)
    neg = N - pos
    pw = neg / pos
    return torch.tensor(pw, dtype=torch.float32)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dataset from all feature subfolders
    dataset = FeatureMultilabelAllDataset(
        features_dir=args.features_dir,
        class_ingredient_path=args.class_ingredient_path,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No features found in {args.features_dir}. Run extract_features.py first.")

    # Infer dims
    sample_feat, sample_label = dataset[0]
    in_dim = sample_feat.shape[0]
    num_labels = sample_label.shape[0]
    print(f"Samples: {len(dataset)} — feat_dim: {in_dim}, num_labels: {num_labels}")

    # Train/val split
    val_count = max(1, int(len(dataset) * args.val_ratio))
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(dataset, [train_count, val_count])
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = SimpleHead(in_features=in_dim, out_features=num_labels, hidden=args.hidden, dropout=args.dropout)
    model = model.to(device)

    # Loss with pos_weight for class imbalance
    mapping = load_class_ingredient_map(args.class_ingredient_path)
    pos_weight = compute_pos_weight(mapping).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for i, (feats, labels) in enumerate(train_loader):
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}  step {i}  loss={loss.item():.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * feats.size(0)
                n_val += feats.size(0)
        val_loss = val_loss / max(1, n_val)
        print(f"[VAL] epoch {epoch+1}  val_loss={val_loss:.4f}")
        model.train()

        # Save best
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, f"allclasses_head.pt")
        torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.checkpoint_dir, f"allclasses_head_best.pt")
            torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, best_path)
            print(f"Saved best checkpoint: {best_path}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default='features')
    parser.add_argument('--class_ingredient_path', type=str, default='data/metadata/class_ingredient.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
