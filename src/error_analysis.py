import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import DetrForObjectDetection, DetrImageProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CKPT = ROOT / 'checkpoints' / 'best_model.pth'
ANN_FILE = DATA / 'annotations' / 'val.json'
IMG_DIR = DATA / 'val'
VIZ = ROOT / 'viz'
VIZ.mkdir(parents=True, exist_ok=True)
(VIZ / 'classification_errors').mkdir(exist_ok=True)
(VIZ / 'localization_errors').mkdir(exist_ok=True)
(VIZ / 'side_by_side').mkdir(exist_ok=True)

MODEL_NAME = 'facebook/detr-resnet-50'
SCORE_THRESHOLD = 0.3
HIGH_CONF_THRESHOLD = 0.5
IOU_CLASS_MATCH = 0.3
IOU_LOC_LOW = 0.1
IOU_LOC_HIGH = 0.5
N_SIDE_BY_SIDE = 6


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(ax, image, boxes, labels, color, names):
    ax.imshow(image)
    for box, lab in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 4), names.get(int(lab), str(lab)),
                color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor='none'))
    ax.set_axis_off()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    id2label = ckpt['id2label']
    cat_id_to_idx = ckpt['cat_id_to_idx']
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
    coco_id_to_name = {c['id']: c['name'] for c in coco_gt.dataset['categories']}
    idx_to_name = {i: coco_id_to_name[cid] for cid, i in cat_id_to_idx.items()}

    class_errors = []
    loc_errors = []
    image_ids = coco_gt.getImgIds()
    side_by_side_saved = 0

    with torch.no_grad():
        for img_id in image_ids:
            info = coco_gt.loadImgs(img_id)[0]
            img = Image.open(IMG_DIR / info['file_name']).convert('RGB')
            inputs = processor(images=img, return_tensors='pt').to(device)
            out = model(**inputs)
            target_size = torch.tensor([[img.height, img.width]], device=device)
            res = processor.post_process_object_detection(
                out, target_sizes=target_size, threshold=SCORE_THRESHOLD
            )[0]

            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            gt_boxes = []
            gt_labels = []
            for a in gt_anns:
                x, y, w, h = a['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(cat_id_to_idx[a['category_id']])

            pred_boxes = res['boxes'].cpu().tolist()
            pred_labels = [int(l.item()) for l in res['labels']]
            pred_scores = [float(s.item()) for s in res['scores']]

            if side_by_side_saved < N_SIDE_BY_SIDE and gt_boxes and pred_boxes:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                draw_boxes(axes[0], img, gt_boxes, gt_labels, 'lime', idx_to_name)
                axes[0].set_title('Ground Truth')
                draw_boxes(axes[1], img, pred_boxes, pred_labels, 'red', idx_to_name)
                axes[1].set_title('Predictions')
                fig.tight_layout()
                fig.savefig(VIZ / 'side_by_side' / f'img_{img_id}.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                side_by_side_saved += 1

            for pbox, plab, pscore in zip(pred_boxes, pred_labels, pred_scores):
                best_iou = 0.0
                best_gt_lab = None
                for gbox, glab in zip(gt_boxes, gt_labels):
                    i = iou_xyxy(pbox, gbox)
                    if i > best_iou:
                        best_iou = i
                        best_gt_lab = glab

                if best_gt_lab is None:
                    continue

                if pscore >= HIGH_CONF_THRESHOLD and best_iou >= IOU_CLASS_MATCH and plab != best_gt_lab:
                    class_errors.append((img_id, pbox, plab, best_gt_lab, pscore, best_iou,
                                         gt_boxes, gt_labels, pred_boxes, pred_labels))
                if plab == best_gt_lab and IOU_LOC_LOW <= best_iou < IOU_LOC_HIGH:
                    loc_errors.append((img_id, pbox, plab, best_gt_lab, pscore, best_iou,
                                       gt_boxes, gt_labels, pred_boxes, pred_labels))

    def save_error(items, subdir, kind):
        for k, item in enumerate(items[:6]):
            img_id, pbox, plab, glab, pscore, iou, gtb, gtl, pb, pl = item
            info = coco_gt.loadImgs(img_id)[0]
            img = Image.open(IMG_DIR / info['file_name']).convert('RGB')
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            draw_boxes(axes[0], img, gtb, gtl, 'lime', idx_to_name)
            axes[0].set_title('Ground Truth')
            draw_boxes(axes[1], img, pb, pl, 'red', idx_to_name)
            pname = idx_to_name.get(plab, str(plab))
            gname = idx_to_name.get(glab, str(glab))
            axes[1].set_title(f'{kind}: pred={pname} gt={gname} score={pscore:.2f} iou={iou:.2f}')
            fig.tight_layout()
            fig.savefig(VIZ / subdir / f'{kind}_{img_id}_{k}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)

    save_error(class_errors, 'classification_errors', 'classification_error')
    save_error(loc_errors, 'localization_errors', 'localization_error')

    summary = {
        'side_by_side_saved': side_by_side_saved,
        'classification_errors_total': len(class_errors),
        'localization_errors_total': len(loc_errors),
        'classification_errors_saved': min(6, len(class_errors)),
        'localization_errors_saved': min(6, len(loc_errors)),
        'thresholds': {
            'score': SCORE_THRESHOLD,
            'high_conf': HIGH_CONF_THRESHOLD,
            'iou_class_match': IOU_CLASS_MATCH,
            'iou_localization_low': IOU_LOC_LOW,
            'iou_localization_high': IOU_LOC_HIGH,
        },
    }
    with open(VIZ / 'error_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
