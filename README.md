# DETR Fine-tuning on a 10-class COCO Subset

Fine-tuning of `facebook/detr-resnet-50` on a 10-class subset of COCO val2017, with TensorBoard logging, a profiler trace, COCO-style evaluation and an error analysis.

## Layout

```
proj/
├── data/
│   ├── train/                 240 images
│   ├── val/                   60 images
│   └── annotations/
│       ├── train.json         COCO-format
│       └── val.json
├── src/
│   ├── train.py
│   ├── eval.py
│   └── error_analysis.py
├── logs/
│   ├── tb/                    TensorBoard events
│   ├── profiler_trace.json    chrome://tracing format
│   ├── metrics.json
│   └── val_predictions.json
├── checkpoints/best_model.pth
└── viz/
    ├── side_by_side/          GT vs Pred
    ├── classification_errors/
    └── localization_errors/
```

## Dataset

10 classes from COCO 2017: `person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, cat, dog`.
The val2017 split was filtered to images containing at least one annotation of those classes and split 80/20:

| Split | Images | Annotations |
|-------|-------:|------------:|
| train | 240    | 1206        |
| val   | 60     | 266         |

## Hyperparameters

| Setting               | Value                  |
|-----------------------|------------------------|
| Base model            | facebook/detr-resnet-50 |
| Epochs                | 8                      |
| Batch size            | 4                      |
| Optimizer             | AdamW                  |
| LR (transformer head) | 1e-4                   |
| LR (ResNet backbone)  | 1e-5                   |
| Weight decay          | 1e-4                   |
| Grad clip (max norm)  | 0.1                    |
| Image processor       | DetrImageProcessor (default normalization, max longest edge 1333) |
| Precision             | fp32                   |
| Hardware              | 1× NVIDIA L4 (24 GB)   |

Classification head and bbox classifier are re-initialized to 10 classes (plus the "no object" class) via `ignore_mismatched_sizes=True`. The remaining DETR weights are loaded from the COCO checkpoint.

## Training Behavior

Per-epoch losses (averaged over batches):

| Epoch | train total | val total |
|-------|------------:|----------:|
| 0     | 2.180       | 1.509     |
| 1     | 1.598       | 1.322     |
| 2     | 1.505       | 1.313     |
| 3     | 1.458       | 1.223     |
| 4     | 1.372       | 1.226     |
| 5     | 1.338       | **1.170** |
| 6     | 1.263       | 1.220     |
| 7     | 1.244       | 1.260     |

Best validation total loss occurs at epoch 5; checkpoint at that epoch is saved to `checkpoints/best_model.pth`. Loss components (classification, bbox L1, GIoU) are logged per batch and per epoch and can be inspected in TensorBoard (`tensorboard --logdir logs/tb`).

A `torch.profiler` trace covering 5 training steps of epoch 0 is exported to `logs/profiler_trace.json` and can be opened in `chrome://tracing/`.

## Results

Computed with `pycocotools` on the val split:

| Metric                 | Value  |
|------------------------|-------:|
| mAP (IoU=0.50:0.95)    | 0.1191 |
| mAP@0.50               | 0.2049 |
| mAP@0.75               | 0.1274 |
| AR@1                   | 0.068  |
| AR@10                  | 0.172  |
| AR@100                 | 0.187  |
| AP small               | 0.073  |
| AP medium              | 0.175  |
| AP large               | 0.274  |

## Error Analysis

Thresholds used by `src/error_analysis.py`:

| Knob                  | Value |
|-----------------------|------:|
| Display score min     | 0.30  |
| "High confidence"     | 0.50  |
| IoU match (class err) | 0.30  |
| IoU window (loc err)  | [0.10, 0.50) |

Counts on the 60-image val set:

| Error type                          | Count |
|-------------------------------------|------:|
| Classification (wrong class, IoU≥0.30, score≥0.50) | 4   |
| Localization  (right class, 0.10 ≤ IoU < 0.50)     | 199 |

### Findings

- **Localization dominates.** With only 8 epochs on 240 images, the model places boxes roughly in the right region of the image and assigns the right class, but the boxes are loose: 199 predictions match a same-class GT with IoU below 0.5. This is consistent with the per-area AP curve (AP_small 0.073 → AP_large 0.274) — small objects in particular suffer from imprecise localization.
- **Classification is largely correct when confidence is high.** Only 4 high-confidence predictions hit a GT box with the wrong label. The remaining classification mistakes tend to be at lower confidence (below the 0.50 cutoff) and are filtered out before reaching the error buckets.
- **Most failures are recall, not category confusion.** AR@100 (0.187) being close to mAP (0.119) implies the bottleneck is not "model proposes the wrong class" but "model proposes too few correct boxes" — DETR's 100 object queries are not yet well-specialized after such a short fine-tune.
- **Confusable categories.** The handful of classification errors are between visually adjacent classes in this subset (e.g., car ↔ truck, person ↔ bicycle when riders overlap).

### Suggested next steps

- Train longer (>=50 epochs) and with cosine LR decay; DETR is famously slow to converge.
- Larger train split (≥ 2k images) or use the full train2017 set for these 10 classes.
- Consider Deformable-DETR for faster convergence and better small-object AP.

## Reproducing

```bash
python src/train.py
python src/eval.py
python src/error_analysis.py
```
