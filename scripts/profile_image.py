#!/usr/bin/env python3
import os, json, requests, torch, sys
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR     = "/mnt/gcs/TwiBot-22"   # directory containing user.json
OUTPUT_FILE  = "profile_image_feats.pt"
BATCH_SIZE   = 128                    # images per CLIP forward pass
MAX_WORKERS  = 100                     # parallel download threads
DEVICE       = torch.device('cuda' if torch.cuda.is_available()
                            else 'mps'  if torch.backends.mps.is_available()
                            else 'cpu')

# ─── LOAD USERS ─────────────────────────────────────────────────────────────
DATA_DIR = Path(DATA_DIR)
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

def load_json_records(fname):
    """Load a JSON file of array- or line- delimited records."""
    path = DATA_DIR / fname
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
    return data

print("Loading user data…")
user_dicts = load_json_records('user.json')
print(f"Loaded {len(user_dicts):,} users from {DATA_DIR/'user.json'}. Now converting to DataFrame…")
users_df = pd.DataFrame(user_dicts)

# ─── EXTRACT IDS & URLS ─────────────────────────────────────────────────────
uids = users_df['id'].astype(str).tolist()
urls = users_df['profile_image_url'].fillna('').tolist()

# ─── MODEL SETUP ────────────────────────────────────────────────────────────
print("Using device:", DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                     .vision_model.to(DEVICE).eval()
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# ─── DOWNLOAD HELPER ─────────────────────────────────────────────────────────
def fetch_image(uid_url):
    uid, url = uid_url
    if not url:
        return uid, None
    try:
        resp = requests.get(url, timeout=5)
        img  = Image.open(BytesIO(resp.content)).convert("RGB")
        return uid, img
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        return uid, None

# ─── STREAMING DOWNLOAD & BATCHED EMBED ─────────────────────────────────────
sum_feats  = {}        # uid → CPU tensor [768]
batch_imgs = []
batch_uids = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    # executor.map streams results as soon as they're ready
    for uid, img in tqdm(exe.map(fetch_image, zip(uids, urls)),
                         total=len(uids),
                         desc="Download&Embed"):
        if img is None:
            continue

        batch_imgs.append(img)
        batch_uids.append(uid)

        if len(batch_imgs) >= BATCH_SIZE:
           # 1) forward under no_grad autocast (CUDA only)
           inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
           with torch.no_grad():
               if DEVICE.type == 'cuda':
                   with torch.cuda.amp.autocast():
                       out = model(**inputs).pooler_output
               else:
                   out = model(**inputs).pooler_output

           # 2) move to CPU & free GPU memory
           feats_cpu = out.cpu()
           del out, inputs
           if DEVICE.type == 'cuda':
               torch.cuda.empty_cache()

           # 3) accumulate on CPU
           for u, f in zip(batch_uids, feats_cpu):
               sum_feats[u] = f

           batch_imgs.clear()
           batch_uids.clear()

# Flush any remaining
if batch_imgs:
    inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
    if DEVICE.type == 'cuda':
        with torch.cuda.amp.autocast():
            feats = model(**inputs).pooler_output
    else:
        feats = model(**inputs).pooler_output
    for u, f in zip(batch_uids, feats.cpu()):
        sum_feats[u] = f

# ─── STACK & SAVE FINAL FEATURES ────────────────────────────────────────────
print("Building final tensor…")
zero = torch.zeros(768)
rows = [ sum_feats.get(uid, zero) for uid in uids ]
tensor = torch.stack(rows, dim=0)
torch.save(tensor, OUTPUT_FILE)
print(f"✅ Saved {OUTPUT_FILE} with shape {tensor.shape}")
