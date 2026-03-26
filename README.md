# Contactless Palmprint Verification (PolyU–IITD)

Deep metric learning model for contactless palmprint verification using ResNet18 + ArcFace angular margin loss, trained on the PolyU–IITD Contactless Palmprint Database v3.0 (611 subjects, 12,220 images).

**Status:** Training complete (val EER 1.13%). Full test-set evaluation and robustness benchmarking coming in Checkpoint 2.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

The training script uses early stopping and saves the best checkpoint based on validation EER.

## Repository Structure

```
src/
  config.py          - Hyperparameters and paths
  dataset.py         - Dataset class with augmentation pipeline
  model.py           - ResNet18 backbone + projection head + ArcFace head
  train.py           - Training loop with early stopping
  create_splits.py   - Subject-disjoint split generation
  corruptions.py     - Image corruption functions (for robustness preview)
splits/              - Subject-disjoint split definitions (JSON)
figures/             - Generated plots and visualizations
checkpoints/         - Saved model weights
requirements.txt     - Python dependencies
report/              - Project report (LaTeX + PDF)
```

## Dataset

PolyU–IITD Contactless Palmprint Database v3.0 with strict subject-disjoint splits:

| Split      | Subjects | Images | Classes |
|------------|----------|--------|---------|
| Train      | 427 (70%)| 8,540  | 854     |
| Validation | 61 (10%) | 1,220  | 122     |
| Test       | 123 (20%)| 2,460  | 246     |
