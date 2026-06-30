# Training Guide — EfficientNet-B4 Deepfake Detector

> **Goal**: Train a reference-free face-swap detector. No enrollment required — detects deepfake artifacts directly.

## Requirements

```bash
# Already in bridge-venv after setup:
pip install timm torchvision albumentations h5py facenet-pytorch wandb
```

**Hardware:**
- Apple M-series (MPS): ~2–4 hours for 10 epochs on ~3k frames ✓ recommended
- Colab A100: ~30–45 min (~$2–3) for faster iteration

---

## Step 1 — Record real face data

Record 2–3 sessions of webcam footage (5–10 min each). Vary angles, lighting, expressions.

```bash
# Session 1 — normal lighting, facing camera
python scripts/record_real_faces.py --session dan_1 --duration 600

# Session 2 — different angle/lighting
python scripts/record_real_faces.py --session dan_2 --duration 600

# Session 3 — glasses, side angles, far distance etc
python scripts/record_real_faces.py --session dan_3 --duration 600
```

Saves ~1 face crop/sec to `data/real/<session>/`. Expect 300–600 crops per session.  
**Target: ≥ 1,000 real crops total** (more = better generalisation).

---

## Step 2 — Generate fake frames

Automatically batch-applies DLC face swap to all real frames with all source faces in a directory.

```bash
python scripts/generate_fakes.py \
    --real-dir data/real \
    --source-dir ~/Desktop/Deep\ Fake\ Tests \
    --dlc ~/PROJECTS/Deep-Live-Cam \
    --max-per-source 600
```

Saves to `data/fake/<source_name>/`. Caps at 600 fakes per source to balance classes.  
With 3 source faces and 1,000 real frames: expect ~1,800 fake crops.

---

## Step 3 — Preprocess to HDF5

Video-level 80/10/10 split (seed=42), packs to compressed HDF5.

```bash
python scripts/preprocess_dataset.py \
    --real-dir data/real \
    --fake-dir data/fake \
    --out data/dataset.h5
```

Output: `data/dataset.h5` (~300–600 MB depending on frame count).

---

## Step 4 — Train

```bash
# Standard run (logs to stdout)
python scripts/train.py --config configs/efficientnet_b4_mvp.yaml

# With W&B logging
WANDB_PROJECT=deepfake-detector python scripts/train.py --config configs/efficientnet_b4_mvp.yaml

# Resume from checkpoint
python scripts/train.py --config configs/efficientnet_b4_mvp.yaml --resume checkpoints/best.pt
```

**Expected output** (per epoch):
```
Epoch 01/10  tr_loss=0.6234 tr_auc=0.7812  va_loss=0.5891 va_auc=0.8234  lr=2.00e-04  142s ← best
Epoch 02/10  tr_loss=0.4912 tr_auc=0.8901  va_loss=0.4234 va_auc=0.9012  lr=1.95e-04  140s ← best
...
```

**Target metrics** (self-generated dataset):
- Val AUC > 0.95 on DLC face-swap (same tool as training)
- Val AUC > 0.80 on unseen tool (FaceFusion, etc.)
- Val AUC < 0.70 on natural pose variation → retrain with more diversity

Checkpoint saved to `checkpoints/best.pt` on every val AUC improvement.

---

## Step 5 — Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/best.pt \
    --model-name efficientnet_b4_deepfake \
    --version v0.1
```

Writes:
- `models/efficientnet_b4_v0.1.onnx` (fp32, ~75 MB)
- `models/efficientnet_b4_v0.1_fp16.onnx` (~38 MB, faster on Apple Silicon)
- SHA256 checksums printed for `models/manifest.json`

---

## Step 6 — Upload to HuggingFace Hub

```bash
pip install huggingface-hub
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='models/efficientnet_b4_v0.1.onnx',
    path_in_repo='efficientnet_b4_v0.1.onnx',
    repo_id='redsitesoftware/deepfake-detector-models',
    repo_type='model',
)
"
```

---

## Option A — Pre-trained model (zero-shot, instant)

DeepfakeBench released an EfficientNet-B4 pre-trained on FF++ — download it right now (67 MB, no form):

```bash
python scripts/download_datasets.py --dfbench
# Downloads: checkpoints/effnb4_ff_pretrained.pth

# Evaluate zero-shot on your HDF5 dataset
python scripts/finetune_pretrained.py \
    --checkpoint checkpoints/effnb4_ff_pretrained.pth \
    --dataset    data/dataset.h5 \
    --eval-only

# Fine-tune on DLC-generated data (recommended for best accuracy)
python scripts/finetune_pretrained.py \
    --checkpoint checkpoints/effnb4_ff_pretrained.pth \
    --dataset    data/dataset.h5 \
    --epochs 20 --freeze-backbone-epochs 5
```

The pre-trained model was trained on FF++ (Deepfakes + FaceSwap + FaceShifter + Face2Face + NeuralTextures at c23 compression). It will partially detect modern DLC-style deepfakes but benefits from fine-tuning on your specific face-swap tool.

---

## Option B — Public datasets (no form required)

```bash
# FF++ real videos via yt-dlp (no approval, ~750 videos available)
python scripts/download_datasets.py --ff-real --ff-limit 100  # start with 100

# DF40 face crops: 40 deepfake methods, Google Drive, no approval (~50 GB)
python scripts/download_datasets.py --df40 --df40-real

# Everything at once
python scripts/download_datasets.py --all-no-approval
```

| Dataset | Access | Size | Face-swap methods | Notes |
|---|---|---|---|---|
| **FF++ real videos** | yt-dlp (public dict) | ~25 GB | — (real only) | ~750/977 available |
| **DF40** | Google Drive (no form) | ~50 GB | ✅ 10 methods (SimSwap, FSGAN, Hififace…) | NeurIPS 2024 |
| **DeepfakeBench effnb4** | GitHub Releases | 67 MB | — (weights, not data) | Pre-trained on FF++ |

## Option C — Form-gated datasets (~2–7 day wait)

| Dataset | Form | Size | Face-swap? |
|---|---|---|---|
| **FaceForensics++** fakes | [bit.ly/faceforensics-form](https://bit.ly/faceforensics-form) | ~10 GB (c23) | ✓ Deepfakes, FaceSwap, FaceShifter |
| **Celeb-DF v2** | [forms.gle/2jYBby6y1FBU3u6q9](https://forms.gle/2jYBby6y1FBU3u6q9) | ~3 GB | ✓ GAN face-swap |
| **DFDC** | AWS IAM required — [ai.meta.com/datasets/dfdc](https://ai.meta.com/datasets/dfdc/) | ~470 GB | ✓ ~5 methods |

**Recommended order**: Start with `--dfbench` (instant, 67 MB) → record your own DLC fakes (Step 1–3) → fine-tune → submit FF++ form for broader generalization.

> **DLC-specific note**: Deep-Live-Cam uses `inswapper_128.onnx` via InsightFace (2024). None of the public datasets include this specific model. Self-generating DLC fakes (Steps 1–3 above) + fine-tuning the pre-trained model gives the best detection accuracy for your deployment scenario.

---

## Integration with detector

After export, update `deepfake_detector/models/registry.py` to point to the new ONNX, and the CNN score will become meaningful for reference-free detection.

---

## Reproduce this training

```bash
git clone https://github.com/redsitesoftware/deepfake-detector
cd deepfake-detector
git checkout feature/issue-12-training
pip install -r requirements.txt
# Then follow Steps 1-5 above
```
