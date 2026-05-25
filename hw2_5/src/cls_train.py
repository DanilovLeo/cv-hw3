import argparse
import json
from collections import Counter
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
CLS_DATA = ROOT / 'cls_data'
LOGS = ROOT / 'logs'
CKPT_DIR = ROOT / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
EPOCHS = 15
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 5e-4


def find_classes(root):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return classes, {c: i for i, c in enumerate(classes)}


class CropDataset(Dataset):
    def __init__(self, roots, class_to_idx, tfm):
        self.samples = []
        self.tfm = tfm
        for root in roots:
            for cls, idx in class_to_idx.items():
                d = root / cls
                if not d.is_dir():
                    continue
                for p in d.iterdir():
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert('RGB')
        return self.tfm(img), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-synth', action='store_true')
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    tag = args.tag or ('augmented' if args.with_synth else 'baseline')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes, class_to_idx = find_classes(CLS_DATA / 'train')
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_roots = [CLS_DATA / 'train']
    if args.with_synth:
        synth_root = CLS_DATA / 'synth'
        if synth_root.exists():
            train_roots.append(synth_root)

    train_set = CropDataset(train_roots, class_to_idx, train_tfm)
    val_set = CropDataset([CLS_DATA / 'val'], class_to_idx, val_tfm)

    label_counts = Counter(y for _, y in train_set.samples)
    weights = torch.tensor(
        [1.0 / max(1, label_counts.get(i, 0)) for i in range(len(classes))],
        dtype=torch.float32,
    )
    sample_weights = [weights[y].item() for _, y in train_set.samples]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=str(LOGS / f'tb_cls_{tag}'))
    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        n = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss /= max(1, n)
        train_acc = correct / max(1, total)

        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(1).cpu().tolist()
                all_y.extend(y.tolist())
                all_p.extend(pred)
        val_acc = sum(a == b for a, b in zip(all_y, all_p)) / max(1, len(all_y))
        val_f1 = f1_score(all_y, all_p, average='macro', zero_division=0)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/macro_f1', val_f1, epoch)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_acc': val_acc, 'val_macro_f1': val_f1})
        print(f'[{tag}] epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} '
              f'val_acc={val_acc:.3f} val_f1={val_f1:.3f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({'model_state': best_state, 'classes': classes,
                    'class_to_idx': class_to_idx},
                   CKPT_DIR / f'cls_{tag}.pth')

    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            all_p.extend(logits.argmax(1).cpu().tolist())
            all_y.extend(y.tolist())

    target_names = [idx_to_class[i] for i in range(len(classes))]
    labels = list(range(len(classes)))
    report = classification_report(all_y, all_p, target_names=target_names,
                                   labels=labels, output_dict=True, zero_division=0)
    print(classification_report(all_y, all_p, target_names=target_names,
                                labels=labels, zero_division=0))

    out = {
        'tag': tag,
        'with_synth': args.with_synth,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'classes': classes,
        'best_val_macro_f1': best_f1,
        'final_val_acc': sum(a == b for a, b in zip(all_y, all_p)) / max(1, len(all_y)),
        'classification_report': report,
        'history': history,
    }
    with open(LOGS / f'cls_metrics_{tag}.json', 'w') as f:
        json.dump(out, f, indent=2)
    writer.close()


if __name__ == '__main__':
    main()
