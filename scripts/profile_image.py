# profile_images.py  — embeds each user’s profile picture instead
import os, torch, requests, json, pandas as pd
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

# Config
USER_JSON   = '/mnt/gcs/TwiBot-22/user.json'
DATA_DIR   = '/mnt/gcs/TwiBot-22'  # GCS mountpoint
OUT_FILE    = 'profile_image_feats.pt'
BATCH_SIZE  = 512
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                     .vision_model.to(DEVICE).eval()
if DEVICE.type=="cuda": torch.backends.cudnn.benchmark = True

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

urls     = users_df['profile_image_url'].tolist()
uids     = users_df['id'].astype(str).tolist()

# Download & preprocess
imgs, uids_buf, feats = [], [], []
sum_feats, counts = {}, {}

for uid, url in tqdm(zip(uids, urls), total=len(uids), desc="Users"):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except:
        continue
    imgs.append(img); uids_buf.append(uid)
    if len(imgs) >= BATCH_SIZE:
        # batch embed
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs).pooler_output  # [B,768]
        for u, f in zip(uids_buf, out):
            sum_feats[u]   = f.cpu()
            counts[u]      = 1
        imgs.clear(); uids_buf.clear()

# flush remainder
if imgs:
    inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs).pooler_output
    for u, f in zip(uids_buf, out):
        sum_feats[u]   = f.cpu()
        counts[u]      = 1

# assemble tensor
ordered  = [str(u) for u in users_df['id']] 
rows     = [ sum_feats.get(u, torch.zeros(768)) for u in ordered ]
tensor   = torch.stack(rows, dim=0)
torch.save(tensor, OUT_FILE)
print("Saved", OUT_FILE, "shape", tensor.shape)
