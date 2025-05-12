#!/usr/bin/env python3
import os, json, torch, requests, pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── CONFIG ────────────────────────────────────────────────────────────────
USER_JSON   = '/mnt/gcs/TwiBot-22/user.json'
OUT_FILE    = 'profile_image_feats.pt'
BATCH_SIZE  = 512            # GPU embed batch size
MAX_WORKERS = 50             # parallel downloader threads
DEVICE      = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available()
                           else 'cpu')

# ─── SETUP ─────────────────────────────────────────────────────────────────
print("Using device:", DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                     .vision_model.to(DEVICE).eval()
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# ─── LOAD USERS ─────────────────────────────────────────────────────────────
print("Loading user data…")
with open(USER_JSON, 'r', encoding='utf-8') as f:
    try:
        users = json.load(f)
    except json.JSONDecodeError:
        f.seek(0)
        users = [json.loads(line) for line in f]

uids = [str(u['id']) for u in users]
urls = [u.get('profile_image_url') for u in users]

# ─── HELPERS ────────────────────────────────────────────────────────────────
def fetch_image(uid_url):
    uid, url = uid_url
    if not url:
        return uid, None
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return uid, img
    except:
        return uid, None

# ─── MAIN: DOWNLOAD + BATCHED EMBED ─────────────────────────────────────────
sum_feats   = {}      # uid → CPU Tensor[768]
batch_imgs  = []
batch_uids  = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    futures = {exe.submit(fetch_image, uv): uv for uv in zip(uids, urls)}

    for future in tqdm(as_completed(futures),
                       total=len(futures),
                       desc="Downloading"):
        uid, img = future.result()
        if img is None:
            continue

        batch_imgs.append(img)
        batch_uids.append(uid)

        if len(batch_imgs) >= BATCH_SIZE:
            # batch preprocess & embed
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            if DEVICE.type == 'cuda':
                # mixed precision on CUDA
                with torch.cuda.amp.autocast():
                    feats = model(**inputs).pooler_output
            else:
                feats = model(**inputs).pooler_output

            # move outputs to CPU & store
            for u, f in zip(batch_uids, feats.cpu()):
                sum_feats[u] = f

            batch_imgs.clear()
            batch_uids.clear()

# flush remainder
if batch_imgs:
    inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
    if DEVICE.type == 'cuda':
        with torch.cuda.amp.autocast():
            feats = model(**inputs).pooler_output
    else:
        feats = model(**inputs).pooler_output

    for u, f in zip(batch_uids, feats.cpu()):
        sum_feats[u] = f

# ─── STACK & SAVE ───────────────────────────────────────────────────────────
print("Building final tensor…")
rows = []
zero = torch.zeros(768)
for uid in uids:
    rows.append(sum_feats.get(uid, zero))
tensor = torch.stack(rows, dim=0)
torch.save(tensor, OUT_FILE)
print(f"✅ Saved {OUT_FILE} with shape {tensor.shape}")
