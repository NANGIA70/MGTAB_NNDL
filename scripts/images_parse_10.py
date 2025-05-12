#!/usr/bin/env python3
import os, glob, json, requests, torch, ijson, pickle
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR             = "/mnt/gcs/TwiBot-22"     # where tweet_*.json and user.json live
SAMPLE_UID_CSV       = "sample_uids_10k.csv"    # 10k user IDs
OUTPUT_FILE          = "image_feats_10k.pt"
CHECKPOINT_FILE      = "image_feats_10k_ckpt.pkl"
CHECKPOINT_INTERVAL  = 50_000                      # checkpoint every 50k images
BATCH_SIZE           = 64                          # images per CLIP forward
MAX_IMAGES_PER_USER  = 20                          # cap per user

# ─── LOAD SAMPLE UIDs ─────────────────────────────────────────────────────
sample_df = pd.read_csv(SAMPLE_UID_CSV, dtype=str)
sample_uids = sample_df['id'].tolist()
num_users = len(sample_uids)
uid2idx = {uid: idx for idx, uid in enumerate(sample_uids)}
print(f"Loaded {num_users} sample user IDs from {SAMPLE_UID_CSV}")

# ─── DEVICE & MODEL ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device).eval()
if device.type=='cuda': torch.backends.cudnn.benchmark = True

# ─── PREALLOCATE STATE ─────────────────────────────────────────────────────
sum_feats = torch.zeros((num_users, 768), device=device)
img_counts = [0]*num_users
processed_images = 0

# ─── RESUME CHECKPOINT ─────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    ck = pickle.load(open(CHECKPOINT_FILE, 'rb'))
    sum_feats.copy_(ck['sum_feats'].to(device))
    img_counts = ck['img_counts']
    processed_images = ck['processed_images']
    print(f"Resumed at {processed_images} images")

# ─── FLUSH BATCH ───────────────────────────────────────────────────────────
def flush_batch(batch_imgs, batch_idxs):
    global processed_images
    if not batch_imgs: return
    # forward
    inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip(**inputs).pooler_output   # [B,768]
    # accumulate
    for idx, f in zip(batch_idxs, feats):
        sum_feats[idx] += f
        img_counts[idx] += 1
        processed_images += 1
    # checkpoint
    if processed_images % CHECKPOINT_INTERVAL < len(batch_imgs):
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump({
                'sum_feats': sum_feats.cpu(),
                'img_counts': img_counts,
                'processed_images': processed_images
            }, f)
        print(f"💾 Checkpoint at {processed_images} images")

# ─── STREAM & PARSE ────────────────────────────────────────────────────────
batch_imgs = []
batch_idxs = []

for fn in tqdm(sorted(glob.glob(os.path.join(DATA_DIR, "tweet_*.json"))), desc="Files"):
    with open(fn, 'r', encoding='utf-8') as f:
        for tw in tqdm(ijson.items(f, 'item'), desc=os.path.basename(fn), leave=False):
            uid = tw.get('author_id')
            if uid not in uid2idx: continue
            idx = uid2idx[uid]
            if img_counts[idx] >= MAX_IMAGES_PER_USER: continue
            # gather media
            media = tw.get('entities', {}).get('media', []) + tw.get('extended_entities', {}).get('media', [])
            for m in media:
                url = m.get('media_url_https') or m.get('media_url') or m.get('url')
                if not url: continue
                try:
                    resp = requests.get(url, timeout=5)
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                except:
                    continue
                batch_imgs.append(img)
                batch_idxs.append(idx)
                if len(batch_imgs) >= BATCH_SIZE:
                    flush_batch(batch_imgs, batch_idxs)
                    batch_imgs, batch_idxs = [], []
                if img_counts[idx] >= MAX_IMAGES_PER_USER:
                    break
# final flush
flush_batch(batch_imgs, batch_idxs)

# ─── AVERAGE & SAVE FINAL ──────────────────────────────────────────────────
print("Averaging and saving...")
avg = torch.zeros_like(sum_feats)
for i in range(num_users):
    c = img_counts[i]
    if c>0: avg[i] = sum_feats[i] / c
# save
torch.save(avg.cpu(), OUTPUT_FILE)
print(f"✅ Saved {OUTPUT_FILE} with shape {avg.shape}")
