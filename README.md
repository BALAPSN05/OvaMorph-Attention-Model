# OvaMorph-Attention-Model

Research prototype for ovarian ultrasound morphology assessment using prototype-guided attention and adaptive experts.

## Current prototype

`OvaMorphNet` implements:

1. EfficientNet-B0 feature extraction
2. Three feature experts: texture, local morphology, and global context
3. Adaptive sample-wise expert routing
4. Learnable morphology prototypes
5. Prototype-guided cross-attention
6. Classification head

> This repository is a research prototype. It does not claim clinical validity or publication novelty.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python train.py --data_dir /path/to/dataset --epochs 10
```

Expected dataset layout:

```text
data_dir/
  class_1/
    image1.png
  class_2/
    image2.png
```

The first implementation supports ordinary `ImageFolder`-style datasets. We will add dataset-specific label mapping after verifying the selected ovarian ultrasound dataset.

## Project structure

```text
src/
  dataset.py
  model.py
  train_utils.py
train.py
requirements.txt
```

## Next milestones

- Verify dataset labels and patient-level split requirements
- Add dataset-specific preprocessing
- Add baseline EfficientNet and ViT experiments
- Add ablation switches for experts, routing, and prototype attention
- Add external validation and statistical evaluation
