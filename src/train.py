import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from transformers import DetrForObjectDetection, DetrImageProcessor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
LOGS = ROOT / 'logs'
CKPT = ROOT / 'checkpoints'
LOGS.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 4
LR = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
MODEL_NAME = 'facebook/detr-resnet-50'


class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, processor):
        with open(ann_file) as f:
            data = json.load(f)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.images = {im['id']: im for im in data['images']}
        self.ann_by_img = {}
        for a in data['annotations']:
            self.ann_by_img.setdefault(a['image_id'], []).append(a)
        self.image_ids = sorted(self.images.keys())
        self.cat_ids = sorted({c['id'] for c in data['categories']})
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.categories = data['categories']

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.images[img_id]
        img = Image.open(self.img_dir / info['file_name']).convert('RGB')
        raw = self.ann_by_img.get(img_id, [])
        remapped = [{**a, 'category_id': self.cat_id_to_idx[a['category_id']]} for a in raw]
        target = {'image_id': img_id, 'annotations': remapped}
        enc = self.processor(images=img, annotations=target, return_tensors='pt')
        return {
            'pixel_values': enc['pixel_values'].squeeze(0),
            'labels': enc['labels'][0],
        }


def make_collate(processor):
    def collate(batch):
        pixel_values = [b['pixel_values'] for b in batch]
        encoding = processor.pad(pixel_values, return_tensors='pt')
        labels = [b['labels'] for b in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels,
        }
    return collate


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

    train_set = CocoDetectionDataset(DATA / 'train', DATA / 'annotations/train.json', processor)
    val_set = CocoDetectionDataset(DATA / 'val', DATA / 'annotations/val.json', processor)

    id2label = {i: next(c['name'] for c in train_set.categories if c['id'] == cid)
                for cid, i in train_set.cat_id_to_idx.items()}
    label2id = {v: k for k, v in id2label.items()}

    collate = make_collate(processor)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate, num_workers=NUM_WORKERS)

    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    param_dicts = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad],
         'lr': LR_BACKBONE},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=LR, weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(log_dir=str(LOGS / 'tb'))
    global_step = 0
    best_val = float('inf')

    prof_sched = schedule(wait=1, warmup=1, active=5, repeat=1)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        record_shapes=False,
        with_stack=False,
    )

    for epoch in range(EPOCHS):
        model.train()
        epoch_total = 0.0
        epoch_ce = 0.0
        epoch_bbox = 0.0
        epoch_giou = 0.0
        n_batches = 0

        if epoch == 0:
            prof.__enter__()

        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = [{k: v.to(device) for k, v in lab.items()} for lab in batch['labels']]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            ld = outputs.loss_dict

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            writer.add_scalar('loss/total', loss.item(), global_step)
            writer.add_scalar('loss/classification', ld['loss_ce'].item(), global_step)
            writer.add_scalar('loss/bbox_l1', ld['loss_bbox'].item(), global_step)
            writer.add_scalar('loss/giou', ld['loss_giou'].item(), global_step)

            epoch_total += loss.item()
            epoch_ce += ld['loss_ce'].item()
            epoch_bbox += ld['loss_bbox'].item()
            epoch_giou += ld['loss_giou'].item()
            n_batches += 1
            global_step += 1

            if epoch == 0:
                prof.step()

        if epoch == 0:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(str(LOGS / 'profiler_trace.json'))
            print(f'profiler trace saved to {LOGS / "profiler_trace.json"}')

        writer.add_scalar('epoch_loss/total', epoch_total / n_batches, epoch)
        writer.add_scalar('epoch_loss/classification', epoch_ce / n_batches, epoch)
        writer.add_scalar('epoch_loss/bbox_l1', epoch_bbox / n_batches, epoch)
        writer.add_scalar('epoch_loss/giou', epoch_giou / n_batches, epoch)

        model.eval()
        val_total = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                pixel_mask = batch['pixel_mask'].to(device)
                labels = [{k: v.to(device) for k, v in lab.items()} for lab in batch['labels']]
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                val_total += outputs.loss.item()
                v_batches += 1
        val_avg = val_total / max(1, v_batches)
        writer.add_scalar('epoch_loss/val_total', val_avg, epoch)

        print(f'epoch {epoch}: train_loss={epoch_total / n_batches:.4f}, val_loss={val_avg:.4f}')

        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                'model_state': model.state_dict(),
                'id2label': id2label,
                'label2id': label2id,
                'cat_id_to_idx': train_set.cat_id_to_idx,
                'idx_to_cat_id': {v: k for k, v in train_set.cat_id_to_idx.items()},
                'epoch': epoch,
                'val_loss': val_avg,
            }, CKPT / 'best_model.pth')

    writer.close()
    print('training complete; best val loss =', best_val)


if __name__ == '__main__':
    main()
