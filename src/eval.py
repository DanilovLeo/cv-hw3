import json
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrForObjectDetection, DetrImageProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CKPT = ROOT / 'checkpoints' / 'best_model.pth'
ANN_FILE = DATA / 'annotations' / 'val.json'
IMG_DIR = DATA / 'val'
MODEL_NAME = 'facebook/detr-resnet-50'
SCORE_THRESHOLD = 0.05


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    id2label = ckpt['id2label']
    idx_to_cat_id = ckpt['idx_to_cat_id']

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    coco_gt = COCO(str(ANN_FILE))
    results = []

    with torch.no_grad():
        for img_id in coco_gt.getImgIds():
            info = coco_gt.loadImgs(img_id)[0]
            img = Image.open(IMG_DIR / info['file_name']).convert('RGB')
            inputs = processor(images=img, return_tensors='pt').to(device)
            out = model(**inputs)
            target_size = torch.tensor([[img.height, img.width]], device=device)
            processed = processor.post_process_object_detection(
                out, target_sizes=target_size, threshold=SCORE_THRESHOLD
            )[0]
            for score, label, box in zip(processed['scores'], processed['labels'], processed['boxes']):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'image_id': img_id,
                    'category_id': idx_to_cat_id[int(label.item())],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score.item()),
                })

    if not results:
        print('no predictions above threshold')
        return

    pred_file = ROOT / 'logs' / 'val_predictions.json'
    with open(pred_file, 'w') as f:
        json.dump(results, f)
    coco_dt = coco_gt.loadRes(str(pred_file))
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    m_ap = ev.stats[0]
    m_ap50 = ev.stats[1]
    m_ap75 = ev.stats[2]

    rows = [
        ('mAP @ IoU=0.50:0.95', m_ap),
        ('mAP @ IoU=0.50', m_ap50),
        ('mAP @ IoU=0.75', m_ap75),
    ]
    width = max(len(r[0]) for r in rows)
    print('\n' + '=' * (width + 16))
    print(f'{"Metric":<{width}} | Value')
    print('-' * (width + 16))
    for name, val in rows:
        print(f'{name:<{width}} | {val:.4f}')
    print('=' * (width + 16))

    metrics_file = ROOT / 'logs' / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({'mAP': m_ap, 'mAP50': m_ap50, 'mAP75': m_ap75}, f, indent=2)


if __name__ == '__main__':
    main()
