import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


ROOT = Path(__file__).resolve().parents[1]
CLS_DATA = ROOT / 'cls_data'
VIZ = ROOT / 'viz' / 'synthetic'
VIZ.mkdir(parents=True, exist_ok=True)

RARE_CLASSES = ['cat', 'traffic_light', 'bus']
N_PER_CLASS = 50
GEN_SIZE = 512
CANNY_LOW = 100
CANNY_HIGH = 200
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5
SEED = 1234

PROMPTS = {
    'cat': 'a high quality photograph of a domestic cat, sharp focus, natural lighting, detailed fur',
    'traffic_light': 'a high quality photograph of a traffic light on a city street, sharp focus, urban scene',
    'bus': 'a high quality photograph of a bus on a city street, sharp focus, daytime',
}
NEG = 'lowres, blurry, deformed, cartoon, painting, watermark, text, signature, duplicate, multiple'


def prep_canny(pil_img):
    img = np.array(pil_img.convert('RGB').resize((GEN_SIZE, GEN_SIZE), Image.BILINEAR))
    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


def make_panel(source, canny, generated, path):
    cell = (GEN_SIZE, GEN_SIZE)
    panel = Image.new('RGB', (cell[0] * 3, cell[1]), 'white')
    panel.paste(source.resize(cell), (0, 0))
    panel.paste(canny, (cell[0], 0))
    panel.paste(generated, (cell[0] * 2, 0))
    panel.save(path)


def main():
    random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny', torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    stats = {}
    panel_budget = 6
    panels_saved = 0
    for cls in RARE_CLASSES:
        src_dir = CLS_DATA / 'train' / cls
        out_dir = CLS_DATA / 'synth' / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        srcs = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        if not srcs:
            print(f'no source crops for {cls}, skipping')
            continue
        prompt = PROMPTS[cls]
        made = 0
        idx = 0
        while made < N_PER_CLASS:
            src_path = srcs[idx % len(srcs)]
            idx += 1
            try:
                source = Image.open(src_path).convert('RGB')
            except Exception:
                continue
            canny = prep_canny(source)
            generator = torch.Generator(device=device).manual_seed(SEED + made + hash(cls) % 100000)
            result = pipe(
                prompt=prompt,
                negative_prompt=NEG,
                image=canny,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]
            out_path = out_dir / f'{cls}_{made:04d}.jpg'
            result.save(out_path, 'JPEG', quality=92)
            if panels_saved < panel_budget:
                make_panel(source, canny, result, VIZ / f'panel_{cls}_{made:02d}.png')
                panels_saved += 1
            made += 1
        stats[cls] = made
        print(f'{cls}: generated {made}')

    with open(CLS_DATA / 'synth_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
