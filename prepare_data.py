import json, os, random, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request

random.seed(0)
SRC = 'annotations/instances_val2017.json'
OUT = Path('data')
KEEP_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'dog', 'cat']

with open(SRC) as f:
    coco = json.load(f)

name_to_id = {c['name']: c['id'] for c in coco['categories']}
keep_ids = {name_to_id[n] for n in KEEP_NAMES}
kept_cats = [c for c in coco['categories'] if c['id'] in keep_ids]

anns_keep = [a for a in coco['annotations'] if a['category_id'] in keep_ids and a.get('iscrowd', 0) == 0]
img_ids_with_kept = sorted({a['image_id'] for a in anns_keep})
random.shuffle(img_ids_with_kept)
img_ids_with_kept = img_ids_with_kept[:300]

train_ids = set(img_ids_with_kept[:240])
val_ids = set(img_ids_with_kept[240:])
id_to_img = {im['id']: im for im in coco['images']}

def build(split_ids):
    imgs = [id_to_img[i] for i in split_ids]
    anns = [a for a in anns_keep if a['image_id'] in split_ids]
    return {'images': imgs, 'annotations': anns, 'categories': kept_cats}

train_json = build(train_ids)
val_json = build(val_ids)

(OUT / 'annotations').mkdir(parents=True, exist_ok=True)
(OUT / 'train').mkdir(parents=True, exist_ok=True)
(OUT / 'val').mkdir(parents=True, exist_ok=True)
with open(OUT / 'annotations/train.json', 'w') as f:
    json.dump(train_json, f)
with open(OUT / 'annotations/val.json', 'w') as f:
    json.dump(val_json, f)

print(f'train: {len(train_json["images"])} imgs, {len(train_json["annotations"])} anns')
print(f'val:   {len(val_json["images"])} imgs, {len(val_json["annotations"])} anns')
print(f'classes: {[c["name"] for c in kept_cats]}')

def fetch(args):
    img, dest_dir = args
    p = Path(dest_dir) / img['file_name']
    if p.exists():
        return
    url = f'http://images.cocodataset.org/val2017/{img["file_name"]}'
    urllib.request.urlretrieve(url, p)

jobs = [(im, OUT / 'train') for im in train_json['images']] + [(im, OUT / 'val') for im in val_json['images']]
with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(fetch, jobs))
print('downloaded', len(jobs), 'images')
