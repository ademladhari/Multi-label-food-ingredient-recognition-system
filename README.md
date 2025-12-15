# Multi-Label Food Ingredient Recognition

A deep learning system for recognizing and predicting ingredient lists from food images using multi-label classification. The model identifies up to 399 different ingredients across 200+ food classes.

## Project Overview

This project builds a **multi-label ingredient recognition system** by:

1. Extracting visual features from food images using a pre-trained ResNet50
2. Training a lightweight neural network head to predict ingredient presence
3. Providing inference to identify ingredients in new food images

## System Architecture

```
Food Image (224×224 RGB)
        ↓
[ResNet50 Backbone] ← Pre-trained on ImageNet (frozen)
        ↓
Feature Vector (2048-dim)
        ↓
[Trainable Head] ← 1024 hidden layer + dropout
        ↓
Raw Logits (399 scores)
        ↓
[Sigmoid Activation]
        ↓
Ingredient Probabilities (0-1 range)
        ↓
[Threshold Filter] ← User-configurable (default: 0.95)
        ↓
Final Ingredient Predictions
```

## Installation

### Requirements

- Python 3.12+
- CUDA 13.0 (for GPU acceleration)
- PyTorch 2.11+ with CUDA support
- GPU: NVIDIA RTX 5060 or compatible (compute capability ≥ 7.0)

### Setup

```bash
# Clone repository
git clone https://github.com/ademladhari/Multi-label-Food-Recognition-ATAI-project.git
cd Multi-label-Food-Recognition-ATAI-project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install torch torchvision numpy pandas tqdm Pillow

# For GPU support (CUDA 13.0):
pip install torch==2.11.0.dev20251215+cu130 torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Project Structure

```
src/
├── extract_features.py          # Feature extraction (ResNet50)
├── model_head.py                # Neural network architecture
├── multilabel_dataset_all.py    # Dataset loader for all classes
├── train_all.py                 # Training script
├── infer_image.py               # Inference on single images
└── utils/
    ├── test_accuracy.py         # Test on dataset samples
    └── test_folder.py           # Batch test on folder of images

data/
├── <food_class_1>/              # Food images organized by class
├── <food_class_2>/
└── metadata/
    ├── class_ingredient.txt      # Ground truth labels (class → ingredients)
    └── ingredient_list_label.txt # Ingredient name mapping

checkpoints/
├── allclasses_head_best.pt      # Best trained model (epoch 8)
└── allclasses_head.pt           # Latest checkpoint

features/
├── <class_1>/                   # Pre-extracted feature vectors (.npy files)
└── <class_2>/
```

## Quick Start

### 1. Feature Extraction (One-time setup)

Extract visual features from all food images using ResNet50:

```bash
python src/extract_features.py --device cuda
```

**Parameters:**

- `--device`: auto|cpu|cuda (default: auto)

**Output:** Features saved to `features/` directory

---

### 2. Training

Train the ingredient classifier on all classes:

```bash
python src/train_all.py \
  --epochs 10 \
  --batch_size 64 \
  --hidden 1024 \
  --dropout 0.2 \
  --lr 0.001
```

**Parameters:**

- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 64)
- `--hidden`: Hidden layer size (default: 1024)
- `--dropout`: Dropout rate (default: 0.2)
- `--lr`: Learning rate (default: 0.001)

**Output:**

- `checkpoints/allclasses_head_best.pt` (best model)
- `checkpoints/allclasses_head.pt` (latest model)

---

### 3. Inference on Single Image

Predict ingredients from a food image:

```bash
python src/infer_image.py \
  --image_path "data/test/image.png" \
  --checkpoint "checkpoints/allclasses_head_best.pt" \
  --ingredient_list "data/metadata/ingredient_list_label.txt" \
  --threshold 0.95
```

**Parameters:**

- `--image_path`: Path to food image
- `--checkpoint`: Path to trained model
- `--ingredient_list`: Path to ingredient name mapping
- `--threshold`: Confidence threshold (0.0-1.0, default: 0.5)

**Output:**

```
Top predictions (threshold: 0.95):
 1. ✓ cucumbers                      prob=0.9977
 2. ✓ feta cheese                    prob=0.9972
 3. ✓ olives                         prob=0.9971
 4. ✓ oregano                        prob=0.9930
 5. ✓ ham                            prob=0.9902

All predictions above 0.95 threshold:
  • cucumbers                      prob=0.9977
  • feta cheese                    prob=0.9972
  • olives                          prob=0.9971
  • oregano                         prob=0.9930
  • ham                             prob=0.9902
  • salad cream                    prob=0.9759
  • olive oil                       prob=0.9554
```

---

## File-by-File Reference

### `extract_features.py`

**Purpose:** Convert raw images to feature vectors  
**Usage:** One-time preprocessing step  
**Key Functions:**

- `select_device()`: Auto-detect CUDA availability
- `iter_image_dirs()`: Recursively find all image folders
- `extract_feature()`: ResNet50 forward pass

### `model_head.py`

**Purpose:** Define neural network architecture  
**Key Class:**

- `SimpleHead(in_features=2048, out_features=399, hidden=1024, dropout=0.2)`

### `multilabel_dataset_all.py`

**Purpose:** Load features and labels for training  
**Key Functions:**

- `load_class_ingredient_map()`: Parse ingredient labels
- `FeatureMultilabelAllDataset`: PyTorch Dataset class

### `train_all.py`

**Purpose:** Train ingredient classifier  
**Key Features:**

- Handles class imbalance with pos_weight
- Validates on 10% of data each epoch
- Saves best model based on validation loss

### `infer_image.py`

**Purpose:** Predict ingredients from images  
**Parameters:**

- Configurable confidence threshold
- Top-K prediction display
- Optional CSV output

### `utils/test_accuracy.py`

**Purpose:** Evaluate accuracy on dataset samples  
**Output:** `test_results.txt` with detailed metrics

### `utils/test_folder.py`

**Purpose:** Batch test all images in a folder  
**Output:** `test_folder_results.txt` with predictions at multiple thresholds

---

## Model Performance

### Training Results (10 epochs)

- **Best Epoch:** 8 (validation loss: 0.1901)
- **Training Data:** 31,257 samples (28,132 train, 3,125 val)
- **Feature Dimension:** 2048 (ResNet50)
- **Output Classes:** 399 ingredients

### Accuracy Metrics (0.95 threshold)

| Dish            | Recall | Predictions | Accuracy  |
| --------------- | ------ | ----------- | --------- |
| Greek Salad     | 75%    | 7/8         | ✓ High    |
| Fried Chicken   | 100%   | 2/2         | ✓ Perfect |
| Bacon & Eggs    | 75%    | 5/4         | ✓ Good    |
| Ham Sandwich    | 100%   | 2/2         | ✓ Perfect |
| Garlic Bread    | 100%   | 4/4         | ✓ Perfect |
| Hot & Sour Soup | 100%   | 6/6         | ✓ Perfect |
| Jambalaya       | 100%   | 4/4         | ✓ Perfect |

### Threshold Recommendations

| Threshold | Use Case                   | Predictions        | False Positives |
| --------- | -------------------------- | ------------------ | --------------- |
| 0.50      | Exploratory, get all ideas | Many (12-23)       | High            |
| 0.80      | Balanced                   | Moderate (8-14)    | Moderate        |
| 0.90      | Good accuracy              | Few (6-10)         | Low             |
| **0.95**  | **Strict, production**     | **Very few (4-7)** | **Minimal**     |

**Recommendation:** Use **0.95-0.97** for best results.

---

## Threshold Explanation

The model outputs probabilities (0-1) for each ingredient. The threshold determines which predictions to keep:

- **Low threshold (0.50):** Keeps many predictions (more false positives)
- **High threshold (0.95+):** Keeps only confident predictions (fewer false positives)

Example for Greek Salad (0.95 threshold):

```
All predictions:      23 ingredients
Filtered (>0.95):     7 ingredients ✓
Correct ingredients:  6/8 found
```

---

## Common Use Cases

### 1. Test on a Single Image

```bash
python src/infer_image.py \
  --image_path "path/to/food.jpg" \
  --checkpoint "checkpoints/allclasses_head_best.pt" \
  --ingredient_list "data/metadata/ingredient_list_label.txt" \
  --threshold 0.95
```

### 2. Batch Test Multiple Images

```bash
python src/utils/test_folder.py
# Tests all images in data/test/ folder
```

### 3. Evaluate Model Accuracy

```bash
python src/utils/test_accuracy.py
# Tests on 8 sample dishes from training set
```

### 4. Retrain with Different Parameters

```bash
python src/train_all.py \
  --epochs 20 \
  --batch_size 32 \
  --hidden 2048 \
  --dropout 0.3 \
  --lr 0.0005
```

---

## Limitations & Future Work

### Current Limitations

- Only works on dishes in training dataset (200+ food classes)
- Unknown dishes will match visually similar known dishes
- Rare ingredients have lower prediction accuracy
- Requires CUDA-capable GPU for reasonable speed

### Future Improvements

- Add more food classes and ingredients
- Fine-tune ResNet50 on food-specific dataset
- Implement attention mechanisms for ingredient localization
- Add data augmentation for better generalization
- Create web API for easy access

---

## Dependencies

```
torch>=2.11.0
torchvision>=0.16.0
numpy
pandas
tqdm
Pillow
```

---

## License

This project is part of ATAI Multi-label Food Recognition course.

---

## Contact

**Author:** Ladhari Adem  
**Repository:** https://github.com/ademladhari/Multi-label-Food-Recognition-ATAI-project  
**Email:** ladhari.adem@example.com

---

## Troubleshooting

### CUDA Error: "no kernel image is available"

**Solution:** Update PyTorch to nightly build with CUDA 13.0 support

```bash
pip install torch==2.11.0.dev20251215+cu130 torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Feature extraction takes too long

**Solution:** Ensure GPU is being used

```bash
python src/extract_features.py --device cuda
```

### Low accuracy on unknown dishes

**Expected:** Model works best on dishes similar to training set. For completely new dishes, use lower threshold (0.80) to see all predictions.

### Git push permission denied

**Solution:** Create your own repository and push there, or ask collaborators to add you.

---

## Changelog

### v1.0 (December 2025)

- ✅ Multi-label ingredient recognition on all 200+ food classes
- ✅ ResNet50 feature extraction
- ✅ Training pipeline with imbalance weighting
- ✅ Inference with configurable threshold
- ✅ Batch testing utilities
- ✅ Achieved 100% recall on well-represented dishes
