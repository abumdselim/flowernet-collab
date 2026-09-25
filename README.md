# FlowerNet — Colab Notebook

Self-contained **Google Colab notebook** for the FlowerNet assignment:
multi-model flower image classification with **5-fold stratified
cross-validation**.

## Files

| File | What it is |
|---|---|
| `FlowerNet_Colab.ipynb` | The full notebook — every step written manually, no external code |
| `flower_dataset.zip` | Custom dataset: 3 flower classes (daisy / rose / sunflower) × 100 images = 300, 224×224 RGB |

## How to run (Google Colab)

1. Open `FlowerNet_Colab.ipynb` in Colab (upload to Drive → open with Colab).
2. **Runtime → Change runtime type → T4 GPU** (full run ≈ 25–40 min).
3. **Runtime → Run all** — when *Step 2* runs, upload `flower_dataset.zip`
   from this repo (download it once to your computer first).
4. All figures are shown inline and saved to `/content/figures`; metrics and
   tables to `/content/results`. The last cell zips everything for download.

## What the notebook does

| Requirement | Coverage |
|---|---|
| Custom dataset, ≥ 3 classes | manual upload → 3 classes × 100 images |
| Cross-validation | `StratifiedKFold(5, shuffle=True, random_state=42)` for every model |
| ≥ 3 models (5 used) | SimpleCNN (scratch) · VGG16 · ResNet50 · MobileNetV2 · EfficientNet-B0 (frozen) |
| Data augmentation | random resized crop / flip / rotation / colour jitter — training folds only |
| Evaluation metrics | accuracy, macro precision/recall/F1, ROC-AUC, out-of-fold confusion matrices |
| Learning curves + overfitting check | epoch curves, accuracy-vs-data-size curve, train–val gap |
