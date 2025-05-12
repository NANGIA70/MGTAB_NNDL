# images_parse_10k_new.py
#!/usr/bin/env python3
import os, glob, pickle, torch, ijson, requests, pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR            = "/mnt/gcs/TwiBot-22"
SAMPLE_UID_CSV      = "sample_uids_10k_new.csv"
OUTPUT_FILE         = "image_feats_10k_new.pt"
CHECKPOINT_FILE     = "image_feats_10k_new_ckpt.pkl"
CHECKPOINT_INTERVAL = 50_000
BATCH_SIZE          = 64
MAX_PER_USER        = 20

# ─── SAMPLE UIDs ───────────────────────────────────────────────────────────
uids = pd.read_csv(SAMPLE_UID_CSV, dtype=str)['id'].tolist()
uid2idx = {u:i for i,u in enumerate(uids)}
num_users = len(uids)
print(f"{num_users} users → image parse")

# ─── MODEL SETUP ────────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device).eval()
if device.type=='cuda': torch.backends.cudnn.benchmark = True

# ─── PREALLOCATE ───────────────────────────────────────────────────────────
sum_feats  = torch.zeros((num_users,768), device=device)
img_counts = [0]*num_users
proc_img   = 0

# ─── RESUME ─────────────────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    ck = pickle.load(open(CHECKPOINT_FILE,'rb'))
    sum_feats.copy_(ck['sum_feats'].to(device))
    img_counts   = ck['img_counts']
    proc_img     = ck['processed']
    print(f"Resumed @ {proc_img} images")

def flush(batch_imgs, batch_idxs):
    global proc_img
    if not batch_imgs: return
    inp = processor(images=batch_imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip(**inp).pooler_output
    for idx, f in zip(batch_idxs, feats):
        sum_feats[idx]  += f
        img_counts[idx]+= 1
        proc_img      += 1
    if proc_img % CHECKPOINT_INTERVAL < len(batch_idxs):
        pickle.dump({'sum_feats':sum_feats.cpu(),
                     'img_counts':img_counts,
                     'processed':proc_img},
                    open(CHECKPOINT_FILE,'wb'))
        print(f"Checkpoint @ {proc_img}")

batch_imgs, batch_idxs = [], []
for fn in tqdm(sorted(glob.glob(os.path.join(DATA_DIR,"tweet_*.json"))), desc="Files"):
    with open(fn,'r') as f:
        for tw in ijson.items(f,'item'):
            uid = tw.get('author_id')
            if uid not in uid2idx: continue
            idx = uid2idx[uid]
            if img_counts[idx] >= MAX_PER_USER: continue
            media = tw.get('entities',{}).get('media',[])+tw.get('extended_entities',{}).get('media',[])
            for m in media:
                url = m.get('media_url_https') or m.get('media_url')
                if not url: continue
                try:
                    r = requests.get(url, timeout=5)
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                except:
                    continue
                batch_imgs.append(img); batch_idxs.append(idx)
                if len(batch_imgs)>=BATCH_SIZE:
                    flush(batch_imgs,batch_idxs)
                    batch_imgs, batch_idxs = [], []
                if img_counts[idx]>=MAX_PER_USER: break
# final
flush(batch_imgs,batch_idxs)

# ─── AVERAGE & SAVE ────────────────────────────────────────────────────────
avg = torch.zeros_like(sum_feats)
for i in range(num_users):
    c = img_counts[i]
    if c>0: avg[i] = sum_feats[i]/c
torch.save(avg.cpu(), OUTPUT_FILE)
print(f"Saved images → {OUTPUT_FILE}  shape={avg.shape}")
