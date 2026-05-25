import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'cls_data'
MIN_SIDE = 8


def build_split(split):
    ann_file = DATA / 'annotations' / f'{split}.json'
    img_dir = DATA / split
    out_dir = OUT / split
    with open(ann_file) as f:
        coco = json.load(f)
    id_to_img = {im['id']: im for im in coco['images']}
    id_to_name = {c['id']: c['name'] for c in coco['categories']}
    counts = Counter()
    for c in coco['categories']:
        (out_dir / c['name'].replace(' ', '_')).mkdir(parents=True, exist_ok=True)
    for k, a in enumerate(coco['annotations']):
        img_info = id_to_img[a['image_id']]
        x, y, w, h = a['bbox']
        if w < MIN_SIDE or h < MIN_SIDE:
            continue
        img = Image.open(img_dir / img_info['file_name']).convert('RGB')
        crop = img.crop((x, y, x + w, y + h))
        cls_name = id_to_name[a['category_id']].replace(' ', '_')
        crop.save(out_dir / cls_name / f'{a["image_id"]}_{a["id"]}.jpg', 'JPEG', quality=92)
        counts[cls_name] += 1
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_counts = build_split('train')
    val_counts = build_split('val')
    summary = {
        'train': dict(train_counts.most_common()),
        'val': dict(val_counts.most_common()),
        'rare_classes_by_train_count': [c for c, _ in train_counts.most_common()[::-1][:3]],
    }
    with open(OUT / 'class_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
