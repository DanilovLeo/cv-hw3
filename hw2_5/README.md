# HW 2.5 — Synthetic Data via Stable Diffusion + ControlNet

Augmenting rare classes of our COCO-subset classification task with Stable-Diffusion-1.5 + ControlNet-Canny, and measuring the effect on a ViT-Tiny classifier.

## Pipeline

1. **`src/build_cls_dataset.py`** — Crop every bbox from the Part 1 detection annotations (min side 8 px) into a folder-per-class classification dataset under `cls_data/{train,val}/<class>/`.
2. **`src/cls_train.py`** — Fine-tune `vit_tiny_patch16_224` from timm. Same architecture and hyperparameters for both runs. `--with-synth` adds `cls_data/synth/` to the training set.
3. **`src/generate_synth.py`** — Use SD-1.5 + ControlNet-Canny to generate 50 synthetic samples per rare class, conditioned on Canny edges extracted from real source crops.

## Dataset

Crops were extracted from the same 10-class COCO subset used in Part 1. Per-class counts (after the 8-px filter):

| Class         | Train | Val | + Synth |
|---------------|------:|----:|--------:|
| person        | 791   | 206 | -       |
| car           | 114   | 12  | -       |
| motorcycle    | 70    | 17  | -       |
| bus           | 28    | 6   | **+50** |
| bicycle       | 26    | 0   | -       |
| truck         | 25    | 1   | -       |
| traffic_light | 23    | 6   | **+50** |
| dog           | 17    | 0   | -       |
| cat           | 15    | 4   | **+50** |
| stop_sign     | 6     | 0   | -       |

Rare classes targeted for augmentation: **`cat`, `traffic_light`, `bus`** — chosen as the three rarest classes that also have ≥3 val crops, so the ablation is measurable. (Classes with zero val crops — bicycle, truck, dog, stop_sign — are kept in the label space but cannot be evaluated here.)

## Generation Details

| Setting              | Value |
|----------------------|-------|
| Base diffusion model | `runwayml/stable-diffusion-v1-5` |
| ControlNet           | `lllyasviel/sd-controlnet-canny` |
| Precision            | fp16 |
| Resolution           | 512 × 512 |
| Sampler              | UniPCMultistepScheduler |
| Inference steps      | 25 |
| Guidance scale       | 7.5 |
| Canny thresholds     | 100 / 200 |
| Per-class budget     | 50 |
| Seed                 | 1234 (per-image offset) |

Prompts:

| Class         | Prompt |
|---------------|--------|
| cat           | "a high quality photograph of a domestic cat, sharp focus, natural lighting, detailed fur" |
| traffic_light | "a high quality photograph of a traffic light on a city street, sharp focus, urban scene" |
| bus           | "a high quality photograph of a bus on a city street, sharp focus, daytime" |

Negative prompt: `lowres, blurry, deformed, cartoon, painting, watermark, text, signature, duplicate, multiple`.

Sample source / Canny edge / generated panels are saved under `viz/synthetic/panel_<class>_NN.png`.

## Classifier Training

| Setting       | Value |
|---------------|-------|
| Backbone      | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (timm, ~5.5 M params, ImageNet-pretrained) |
| Epochs        | 15 |
| Batch size    | 32 |
| Optimizer     | AdamW |
| LR            | 3e-4 |
| Weight decay  | 5e-4 |
| Sampler       | `WeightedRandomSampler` (inverse class frequency) |
| Augmentations | Resize 224, RandomHorizontalFlip, ColorJitter(0.2) |
| Best checkpoint | selected by val macro-F1 |

Both runs use **identical** hyperparameters; only the train set differs.

## Ablation Results

Evaluated on the same val crops (252 samples, 7 classes with non-zero support):

| Metric              | Baseline | + Synth | Δ |
|---------------------|---------:|--------:|---:|
| Val accuracy        | 0.905    | 0.921   | **+0.016** |
| Macro F1 (10 cls)   | 0.41     | 0.47    | **+0.06**  |
| Weighted F1         | 0.90     | 0.92    | +0.02 |

### Per-class F1 on classes with val support

| Class         | Support | Baseline F1 | + Synth F1 | Δ        |
|---------------|--------:|------------:|-----------:|---------:|
| **cat**           | 4   | 0.40 | **0.86** | **+0.46** |
| **bus**           | 6   | 0.77 | **0.92** | **+0.15** |
| **traffic_light** | 6   | 0.44 | **0.60** | **+0.16** |
| person        | 206 | 0.95 | 0.97 | +0.02 |
| car           | 12  | 0.71 | 0.73 | +0.02 |
| motorcycle    | 17  | 0.79 | 0.62 | -0.17 |
| truck         | 1   | 0.00 | 0.00 | 0.00 |

## Findings

- **Synthetic augmentation produced large gains on the three targeted rare classes.** The biggest jump is on `cat` (F1 0.40 → 0.86) — the baseline classifier was recalling only 1/4 cats correctly; with +50 synthetic cats, it recalls 3/4.
- **`traffic_light` and `bus` also improved cleanly** (+0.16 and +0.15 F1 respectively), with recall jumping from 0.33 → 0.50 on traffic lights and 0.83 → 1.00 on buses.
- **One regression: `motorcycle`** dropped from 0.79 → 0.62 F1. Likely cause: the augmented model has shifted its decision boundary toward "bus" and "bicycle" (visually similar urban classes), causing some motorcycles to be misrouted. This is a common cost of synthetic augmentation — boosting one rare class can suppress visually adjacent classes.
- **Net macro-F1 still improves +0.06** despite the motorcycle regression, indicating that the synthetic-data gains on rare classes outweigh the secondary regression on a middle-frequency class.
- **Caveats.** Val is small (252 samples, 4–17 per rare class), so the per-class deltas have meaningful variance. Per-class F1 swings of ±0.10 on classes with <10 support samples should not be over-interpreted. A larger val split would be needed to firm up these conclusions.

## Reproducing

```bash
python src/build_cls_dataset.py
python src/cls_train.py                 # baseline
python src/generate_synth.py            # ~10 min on L4, downloads ~7 GB once
python src/cls_train.py --with-synth    # augmented
```
