# README.md

# Robust Contactless Palmprint Verification (PolyU–IITD) + Robustness Benchmark

This project trains a **contactless palmprint verification** model and produces a **robustness report** showing how verification performance changes under real-world capture variations (rotation, scale, lighting, blur, compression, occlusion). It also includes **one improvement** (robust augmentation training or quality gating) and compares results.

## What you get
- A reproducible **subject-disjoint** verification pipeline
- Baseline verifier (embedding model + cosine similarity)
- Robustness suite with severity levels + plots
- Optional improvement and clean vs corrupted comparison

---

## 1) Requirements
- Python 3.10+ (3.11 recommended)
- macOS/Linux/Windows
- GPU recommended but not required for small runs

Install:
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

