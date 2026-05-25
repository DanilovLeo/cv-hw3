# ДЗ 2.5 — Синтетические данные через Stable Diffusion + ControlNet

Аугментация редких классов с помощью SD-1.5 + ControlNet-Canny и измерение эффекта на классификаторе ViT-Tiny.

## Структура

```
hw2_5/
├── src/
│   ├── build_cls_dataset.py   # нарезка боксов из аннотаций ДЗ 2
│   ├── generate_synth.py      # генерация через SD + ControlNet
│   └── cls_train.py           # обучение ViT-Tiny (--with-synth для аугментированного)
├── logs/
│   ├── tb_cls_baseline/
│   ├── tb_cls_augmented/
│   ├── cls_metrics_{baseline,augmented}.json
│   └── *.log
└── viz/synthetic/             # панели: исходник / Canny / результат
```

> `cls_data/` исключён из репозитория — сгенерируется при запуске `build_cls_dataset.py`.

## Датасет

Кропы из аннотаций ДЗ 2 (фильтр: мин. сторона 8 пx). Редкие классы для аугментации: **cat, traffic_light, bus** — единственные с ≥3 val-примерами.

| Класс         | Train | Val | + Synth |
|---------------|------:|----:|--------:|
| person        | 791   | 206 | —       |
| car           | 114   | 12  | —       |
| motorcycle    | 70    | 17  | —       |
| bus           | 28    | 6   | **+50** |
| bicycle       | 26    | 0   | —       |
| truck         | 25    | 1   | —       |
| traffic_light | 23    | 6   | **+50** |
| dog           | 17    | 0   | —       |
| cat           | 15    | 4   | **+50** |
| stop_sign     | 6     | 0   | —       |

## Генерация

| Параметр            | Значение                          |
|---------------------|-----------------------------------|
| Модель              | runwayml/stable-diffusion-v1-5    |
| ControlNet          | lllyasviel/sd-controlnet-canny    |
| Точность            | fp16, 512×512                     |
| Шаги / CFG          | 25 / 7.5                          |
| Пороги Canny        | 100 / 200                         |
| На класс            | 50 изображений                    |

Промпты: `"a high quality photograph of a <class>, sharp focus, natural lighting"`. Негативный промпт: `lowres, blurry, deformed, cartoon, watermark`.

## Классификатор

| Параметр      | Значение                                         |
|---------------|--------------------------------------------------|
| Backbone      | vit_tiny_patch16_224 (timm, ~5.5M, ImageNet)     |
| Эпох / батч   | 15 / 32                                          |
| Оптимизатор   | AdamW, LR 3e-4, WD 5e-4                          |
| Сэмплер       | WeightedRandomSampler (обратная частота классов) |
| Чекпойнт      | лучший val macro-F1                              |

Оба запуска — идентичные гиперпараметры, отличается только тренировочный набор.

## Результаты ablation

| Метрика           | Baseline | + Synth | Δ          |
|-------------------|---------:|--------:|:----------:|
| Val accuracy      | 0.905    | 0.921   | **+0.016** |
| Macro F1          | 0.41     | 0.47    | **+0.06**  |
| Weighted F1       | 0.90     | 0.92    | +0.02      |

### F1 по классам (val)

| Класс             | Support | Baseline | + Synth | Δ         |
|-------------------|--------:|---------:|--------:|----------:|
| **cat**           | 4       | 0.40     | 0.86    | **+0.46** |
| **bus**           | 6       | 0.77     | 0.92    | **+0.15** |
| **traffic_light** | 6       | 0.44     | 0.60    | **+0.16** |
| person            | 206     | 0.95     | 0.97    | +0.02     |
| car               | 12      | 0.71     | 0.73    | +0.02     |
| motorcycle        | 17      | 0.79     | 0.62    | −0.17     |
| truck             | 1       | 0.00     | 0.00    | 0.00      |

**Выводы:**
- Наибольший прирост на целевых редких классах: cat +0.46, traffic_light +0.16, bus +0.15.
- Регрессия на motorcycle (−0.17): модель сместила границы решений в сторону визуально схожих bus/bicycle.
- Несмотря на регрессию, macro-F1 вырос на +0.06 — эффект синтетики на редких классах перевешивает.
- Val выборка мала (4–17 примеров на редкий класс), поэтому дельты следует воспринимать как ориентир.

## Воспроизведение

```bash
python src/build_cls_dataset.py
python src/cls_train.py                 # baseline
python src/generate_synth.py            # ~10 мин на L4, ~7 ГБ загрузок
python src/cls_train.py --with-synth    # с синтетикой
```
