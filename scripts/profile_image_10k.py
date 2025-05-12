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
DATA_DIR      = "/mnt/gcs/TwiBot-22"       # directory containing user.json
SAMPLE_UID_CSV= "sample_uids_10k.csv"      # 10k user IDs
OUTPUT_FILE   = "profile_image_feats_10k.pt"
BATCH_SIZE    = 256                          # images per CLIP forward pass
MAX_WORKERS   = 50                           # parallel download threads
DEVICE        = torch.device('cuda' if torch.cuda.is_available() 
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
print(f"Loaded {len(user_dicts):,} users. Filtering to sample of 10k…")

# ─── FILTER TO SAMPLE UIDS ───────────────────────────────────────────────────
sample_df = pd.read_csv(SAMPLE_UID_CSV, dtype=str)
sample_uids = set(sample_df['id'].tolist())
filtered = [u for u in user_dicts if str(u.get('id')) in sample_uids]
users_df = pd.DataFrame(filtered)
print(f"Filtered down to {len(users_df):,} users.")

# ─── EXTRACT IDS & PROFILE URLS ──────────────────────────────────────────────
uids = users_df['id'].astype(str).tolist()
urls = users_df['profile_image_url'].fillna('').tolist()
num_users = len(uids)
uid2idx = {uid: i for i, uid in enumerate(uids)}

# ─── MODEL SETUP ────────────────────────────────────────────────────────────
print("Using device:", DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                     .vision_model.to(DEVICE).eval()
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# ─── PREALLOCATE OUTPUT TENSOR ──────────────────────────────────────────────
image_feats = torch.zeros((num_users, 768), device=DEVICE)

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
batch_imgs, batch_idxs = [], []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    for uid, img in tqdm(exe.map(fetch_image, zip(uids, urls)),
                         total=num_users,
                         desc="Download & Embed"):
        if img is None:
            continue
        batch_imgs.append(img)
        batch_idxs.append(uid2idx[uid])
        if len(batch_imgs) >= BATCH_SIZE:
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            # mixed precision on CUDA
            with torch.no_grad():
                if DEVICE.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        feats = model(**inputs).pooler_output
                else:
                    feats = model(**inputs).pooler_output
            image_feats[batch_idxs] = feats
            batch_imgs.clear()
            batch_idxs.clear()
# flush remainder
if batch_imgs:
    inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                feats = model(**inputs).pooler_output
        else:
            feats = model(**inputs).pooler_output
    image_feats[batch_idxs] = feats

# ─── SAVE TO DISK ────────────────────────────────────────────────────────────
print("Saving final tensor…")
torch.save(image_feats.cpu(), OUTPUT_FILE)
print(f"✅ Saved {OUTPUT_FILE} with shape {image_feats.shape}")
