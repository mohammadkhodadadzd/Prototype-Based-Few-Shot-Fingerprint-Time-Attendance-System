# Prototype-Based-Few-Shot-Fingerprint-Time-Attendance-System
A lightweight, real-time fingerprint recognition system that uses a triplet-trained ResNet-18 embedding network and few-shot prototype matching to reliably track employee time-attendance with minimal enrollment samples.
## 📈 Performance Results

### Real-Data Identification Results

| File       | True | Pred | Sim  |  ✓/✕ |
| ---------- | ---- | ---- | ---- | :--: |
| 00000.bmp  |  0   |  0   | 0.97 |  ✅  |
| 00001.bmp  |  1   |  1   | 0.81 |  ✅  |
| 00002.bmp  |  2   |  2   | 0.96 |  ✅  |
| 00003.bmp  |  3   |  3   | 0.96 |  ✅  |
| 00004.bmp  |  4   |  4   | 0.89 |  ✅  |
| 00005.bmp  |  5   |  5   | 0.91 |  ✅  |
| 00006.bmp  |  6   |  6   | 0.96 |  ✅  |
| 00007.bmp  |  7   |  5   | 0.96 |  ❌  |
| 00008.bmp  |  8   |  8   | 0.96 |  ✅  |
| 00009.bmp  |  9   |  9   | 0.97 |  ✅  |

**Real-Data Accuracy:** 9/10 = **0.900**

---

### Test Set Accuracy

**Test Set Accuracy:** 91/120 = **0.758**

---

## ⚙️ Hyperparameters

- **BATCH_SIZE:** 32  
- **EMB_DIM:** 128  
- **LR:** 1e-4  
- **MARGIN:** 0.6  
- **EPOCHS:** 15  
- **PATIENCE:** 3  
- **K_SHOT:** 5  

---

## 🚀 Quick Start

1. **Install dependencies**  
   ```bash
   python train_triplet.py \
  --data-dir ./dataset_FVC2000_DB4_B/train_data \
  --epochs 15 \
  --batch-size 32 \
  --margin 0.6 \
  --embedding-dim 128 \
  --lr 1e-4pip install torch torchvision matplotlib
  from system import FingerprintSystem
system = FingerprintSystem(
  encoder_path='best_resnet18.pth',
  transform=eval_transform,
  device=DEVICE
)
# Enroll 5 samples for user 3:
system.enroll(3, [
  'train_data/00003_00.bmp',
  'train_data/00003_01.bmp',
  'train_data/00003_02.bmp',
  'train_data/00003_03.bmp',
  'train_data/00003_04.bmp'
])
pred, score = system.identify('real_data/00003.bmp', threshold=0.8)
print(f"Predicted: {pred}, Similarity: {score:.2f}")
