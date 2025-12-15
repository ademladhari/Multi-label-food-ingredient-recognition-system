import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_class_ingredient_map(mapping_path):
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_name = parts[0]
            vec = np.array([int(x) for x in parts[1:]], dtype=np.float32)
            mapping[class_name] = vec
    return mapping


class FeatureMultilabelAllDataset(Dataset):
    """Loads features for ALL classes from a features root dir.
    Expects layout: features/<class_name>/*.npy
    Labels are taken from class_ingredient.txt mapping by class_name.
    """

    def __init__(self, features_dir: str, class_ingredient_path: str, transform=None, include_classes=None):
        self.features_dir = features_dir
        self.transform = transform

        mapping = load_class_ingredient_map(class_ingredient_path)
        self.mapping = mapping

        # Discover class folders under features_dir
        class_dirs = []
        for name in os.listdir(features_dir):
            path = os.path.join(features_dir, name)
            if not os.path.isdir(path):
                continue
            if include_classes and name not in include_classes:
                continue
            class_dirs.append((name, path))

        self.samples = []  # list of (feature_path, label_vector)
        n_missing = 0
        for class_name, cdir in sorted(class_dirs):
            if class_name not in mapping:
                n_missing += 1
                continue
            files = sorted(f for f in os.listdir(cdir) if f.lower().endswith('.npy'))
            label = mapping[class_name]
            for f in files:
                self.samples.append((os.path.join(cdir, f), label))

        if n_missing > 0:
            print(f"[FeatureMultilabelAllDataset] Skipped {n_missing} classes not in mapping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        feat = np.load(feat_path)
        if self.transform is not None:
            feat = self.transform(feat)
        feat = torch.from_numpy(feat).float()
        label_t = torch.from_numpy(label).float()
        return feat, label_t
