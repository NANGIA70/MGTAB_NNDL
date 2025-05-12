# profile_image_10k_new.py
#!/usr/bin/env python3
import os, json, requests, torch, sys, pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR       = "/mnt/gcs/TwiBot-22"
SAMPLE_UID_CSV = "sample_uids_10k_new.csv"
OUTPUT_FILE    = "profile_image_feats_10k_new.pt"
BATCH_SIZE     = 256
MAX_WORKERS    = 50
DEVICE         = torch.device('cuda' if torch.cuda.is_available()
                            else 'mps'  if torch.backends.mps.is_available()
                            else 'cpu')

# # ─── LOAD & FILTER USERS ────────────────────────────────────────────────────
# DATA_DIR = Path(DATA_DIR)
# def load_json_records(fname):
#     p = DATA_DIR/fname
#     with open(p,'r',encoding='utf-8') as f:
#         try: d=json.load(f)
#         except: 
#             f.seek(0)
#             d=[json.loads(l) for l in f]
#     return d

# print("Loading user data…")
# users = load_json_records("user.json")
# uids  = pd.read_csv(SAMPLE_UID_CSV,dtype=str)['id'].tolist()
# uids  = list(filter(lambda u: str(u) in set(uids), [u['id'] for u in users]))
# urls  = {str(u['id']):u.get('profile_image_url','') for u in users}

# uid2idx = {u:i for i,u in enumerate(uids)}
# num_users = len(uids)

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

# ─── MODEL ─────────────────────────────────────────────────────────────────
print("Using device:", DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                     .vision_model.to(DEVICE).eval()
if DEVICE.type=='cuda': torch.backends.cudnn.benchmark=True

# ─── PREALLOCATE & FETCH ───────────────────────────────────────────────────
image_feats = torch.zeros((num_users,768), device=DEVICE)
def fetch(uid):
    url = urls.get(str(uid),'')
    if not url: return uid, None
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return uid, img
    except:
        return uid, None

# ─── STREAM & EMBED ────────────────────────────────────────────────────────
print("Processing images…")
batch_imgs, batch_idxs = [], []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    for uid,img in tqdm(exe.map(fetch, uids), total=num_users, desc="Profile"):
        if img is None: continue
        batch_imgs.append(img); batch_idxs.append(uid2idx[str(uid)])
        if len(batch_imgs)>=BATCH_SIZE:
            inp = processor(images=batch_imgs,return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                if DEVICE.type=='cuda':
                    with torch.cuda.amp.autocast(): feats=model(**inp).pooler_output
                else:
                    feats=model(**inp).pooler_output
            image_feats[batch_idxs] = feats
            batch_imgs, batch_idxs = [], []
# final
if batch_imgs:
    inp = processor(images=batch_imgs,return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model(**inp).pooler_output
    image_feats[batch_idxs] = feats

# ─── SAVE ──────────────────────────────────────────────────────────────────
torch.save(image_feats.cpu(), OUTPUT_FILE)
print(f"Saved profile → {OUTPUT_FILE}  shape={image_feats.shape}")
