#!/usr/bin/env python3
import os, glob, asyncio, aiohttp, torch, ijson
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from pathlib import Path
import json
import requests

# ─── CONFIG ────────────────────────────────────────────────────────────────
TWEET_JSON_GLOB   = '../Dataset/TwiBot-22/tweet_*.json'
USER_JSON         = '../Dataset/TwiBot-22/user.json'
DATA_DIR         = '../Dataset/TwiBot-22'
OUTPUT_FILE       = 'image_feats.pt'
BATCH_SIZE        = 64       # GPU embed batch size (reduced to avoid OOM)
MAX_PER_USER      = 5       # max images per user
MAX_CONCURRENT    = 100      # parallel HTTP sessions
MAX_IMAGES_PER_USER = 5      # cap per user
CHECKPOINT_INTVL    = 100_000 # image‐level checkpoint interval
CHECKPOINT_FILE     = "image_feats_checkpoint.pt"

# ─── DEVICE ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ─── MODEL ─────────────────────────────────────────────────────────────────
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                     .vision_model.to(device).eval()
if device.type=="cuda":
    torch.backends.cudnn.benchmark = True

# ─── STATE ─────────────────────────────────────────────────────────────────
sum_feats     = {}   # uid -> CPU Tensor[768]
img_counts    = {}   # uid -> int count
buffer_imgs   = []
buffer_uids   = []
processed_imgs = 0

# ─── AVERAGE & SAVE FINAL TENSOR ───────────────────────────────────────────
# load ordered user list (must match your tweets/user parse)
# load users
DATA_DIR = Path(DATA_DIR)
# make sure the data directory exists
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

def load_json_records(fname):
    """Load a JSON file of array- or line- delimited records."""
    path = DATA_DIR / fname
    with open(path, 'r', encoding='utf-8') as f:
        # if the file is a single large JSON array:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # fallback: one JSON object per line
            f.seek(0)
            data = [json.loads(line) for line in f]
    return data

print("Loading user data…")
user_dicts = load_json_records('user.json')

print(f"Loaded {len(user_dicts):,} users from {DATA_DIR / 'user.json'}. Now coverting to DataFrame…")
users_df = pd.DataFrame(user_dicts)

ordered_uids = users_df['id'].astype(str).tolist()

# ─── CHECKPOINT HELPERS ─────────────────────────────────────────────────────
def save_checkpoint():
    cpu_feats = {uid:feat.cpu() for uid,feat in sum_feats.items()}
    torch.save({
        "sum_feats":      cpu_feats,
        "img_counts":     img_counts,
        "processed_imgs": processed_imgs
    }, CHECKPOINT_FILE)
    print(f"💾 Checkpoint @ {processed_imgs} images")

# resume if exists
if os.path.exists(CHECKPOINT_FILE):
    ck = torch.load(CHECKPOINT_FILE, map_location="cpu")
    sum_feats      = {uid:feat.to(device) for uid,feat in ck["sum_feats"].items()}
    img_counts     = ck["img_counts"]
    processed_imgs = ck["processed_imgs"]
    print(f"▶ Resumed from {processed_imgs} images")

# ─── EMBED + ACCUMULATE ────────────────────────────────────────────────────
@torch.no_grad()
def embed_and_accumulate():
    global buffer_imgs, buffer_uids, processed_imgs
    imgs, uids = buffer_imgs, buffer_uids
    buffer_imgs, buffer_uids = [], []

    # 1) forward (autocast only on CUDA)
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            feats = model(**inputs).pooler_output
    else:
        feats = model(**inputs).pooler_output

    # 2) to CPU and free GPU/MPS
    feats_cpu = feats.cpu()
    del feats, inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3) accumulate on CPU
    for uid, feat in zip(uids, feats_cpu):
        if img_counts.get(uid, 0) >= MAX_IMAGES_PER_USER:
            continue
        img_counts[uid] = img_counts.get(uid, 0) + 1
        if uid not in sum_feats:
            sum_feats[uid] = torch.zeros(768)
        sum_feats[uid] += feat
        processed_imgs += 1

    # 4) checkpoint
    if processed_imgs % CHECKPOINT_INTVL < BATCH_SIZE:
        save_checkpoint()

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────
for fn in tqdm(sorted(glob.glob(os.path.join(DATA_DIR, "tweet_*.json"))), desc="Files"):
    with open(fn, "r", encoding="utf-8") as f:
        for tw in tqdm(ijson.items(f, "item"),
                       desc=os.path.basename(fn), leave=False, unit="imgs"):
            uid = tw.get("author_id")
            if img_counts.get(uid, 0) >= MAX_IMAGES_PER_USER:
                continue

            media = tw.get("entities",{}).get("media",[]) + tw.get("extended_entities",{}).get("media",[])
            for m in media:
                url = m.get("media_url_https") or m.get("media_url")
                if not url:
                    continue
                try:
                    resp = requests.get(url, timeout=5)
                    img  = Image.open(BytesIO(resp.content)).convert("RGB")
                except:
                    continue

                buffer_imgs.append(img)
                buffer_uids.append(uid)
                if len(buffer_imgs) >= BATCH_SIZE:
                    embed_and_accumulate()
                if img_counts.get(uid, 0) >= MAX_IMAGES_PER_USER:
                    break

# flush remainder
if buffer_imgs:
    embed_and_accumulate()
save_checkpoint()

rows = []
for uid in ordered_uids:
    cnt = img_counts.get(uid, 0)
    if cnt > 0:
        rows.append(sum_feats[uid] / cnt)
    else:
        rows.append(torch.zeros(768))
image_feats = torch.stack(rows, dim=0)
torch.save(image_feats, OUTPUT_FILE)
print(f"✅ Saved {OUTPUT_FILE}: {image_feats.shape}")