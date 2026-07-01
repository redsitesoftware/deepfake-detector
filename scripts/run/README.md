# Deepfake Detector — Run Scripts

These scripts are in `scripts/run/`. Run them in order from the project root:

```
cd /tmp/deepfake-detector
```

---

## Scripts

| Script | Purpose | When to run |
|--------|---------|-------------|
| `1_fix_venv.sh` | Repair bridge venv (PySide6 broke deps) | Once, before anything |
| `2_train.sh` | Fine-tune EfficientNet-B4 on DF40 | Overnight, alone |
| `3_export_onnx.sh` | Export best checkpoint → ONNX | After AUC > 0.75 |
| `4_webserver.sh` | Start web UI at localhost:8765 | After training done |
| `5_watch_training.sh` | Monitor training log live | While 2_train.sh runs |
| `6_stop_all.sh` | Kill all background processes | Any time |

---

## Step-by-step

### Step 1 — Fix venv (run once)
```bash
bash scripts/run/1_fix_venv.sh
```

### Step 2 — Train (run overnight, web server must be OFF)
```bash
bash scripts/run/2_train.sh

# Monitor in another Terminal:
bash scripts/run/5_watch_training.sh
```

Training stats:
- Start: epoch 1 checkpoint, val AUC = 0.538
- Speed: ~1.5 batch/s → ~45 min/epoch → ~11 hrs for 15 epochs
- Target: val AUC > 0.75 by epoch 10

Stop training: `bash scripts/run/6_stop_all.sh`

### Step 3 — Export ONNX (after AUC > 0.75)
```bash
bash scripts/run/3_export_onnx.sh
```
Writes to `models/efficientnet_b4_deepfake_v1.0.onnx`

### Step 4 — Start web UI
```bash
bash scripts/run/4_webserver.sh
# Opens http://localhost:8765 automatically
```

Web UI usage:
1. Upload a **clear, front-facing JPEG** of someone (celebrity, test face, etc.)
2. Allow webcam access in browser
3. Left panel = your raw cam, Middle = DLC face-swapped, Right = detection scores
4. Press **Space** to reset temporal baseline
5. Press **D** to toggle triple/dual panel view

---

## Current state

- ✅ Best checkpoint: `checkpoints/effnb4_finetuned_best.pt` (epoch 1, val AUC=0.538)
- ✅ Web UI fully working (DLC swap confirmed, WebSocket stable)
- ⏳ CNN detection score shows `—` until ONNX model exported
- ⏳ Training needs restart with fixed hyperparameters (see Step 2)

---

## Key paths

| Path | Description |
|------|-------------|
| `/tmp/bridge-venv` | Python venv (use for all scripts — NOT DLC venv) |
| `/tmp/deepfake-detector` | Project root |
| `checkpoints/effnb4_finetuned_best.pt` | Best trained model |
| `checkpoints/effnb4_ff_pretrained.pth` | DeepfakeBench baseline (67MB) |
| `data/df40` | DF40 dataset (real + fake frames) |
| `/tmp/finetune_log2.txt` | Training log |
| `/tmp/dd_server.log` | Web server log |

---

## Troubleshooting

**"No face detected" on upload**
→ Use a clear, front-facing JPEG (not RGBA PNG, not profile shot)

**CNN score shows `—` in web UI**
→ ONNX model not yet exported. Run Step 3 after training.

**Training crashes immediately**
→ Run `bash scripts/run/1_fix_venv.sh` first to fix numpy/h5py

**Web server hangs on face swap**
→ Check `/tmp/dd_server.log` for errors. DLC needs PySide6 (included in Step 1)

**Training very slow (< 0.5 batch/s)**
→ Check if web server is running (they fight over MPS). Kill it first.

---

## Architecture notes

- **CNN**: EfficientNet-B4 (`efficientnet_pytorch` lib, NOT timm)
  - Class 0 = real, Class 1 = fake
  - Input: 380×380, ImageNet normalised
  - Checkpoint format: DeepfakeBench (`backbone.efficientnet.*`)

- **DLC integration**: `Deep-Live-Cam/modules/processors/frame/face_swapper.py`
  - Key fn: `process_frame(source_face, frame)`
  - Uses `inswapper_128_coreml.onnx` on Apple Neural Engine

- **Web UI**: FastAPI + WebSocket + vanilla JS
  - Route: `POST /api/source` (upload face image)
  - Route: `WS /ws/live/{session_id}` (stream frames)
  - Static: `api/static/index.html` + `api/static/app.js`
