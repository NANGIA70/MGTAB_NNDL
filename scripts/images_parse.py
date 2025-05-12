# images_parse.py

import os, glob, pickle, torch, ijson, requests, json
from tqdm.auto import tqdm
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR               = "/mnt/gcs/TwiBot-22"            # where tweet_*.json lives
USER_JSON              = os.path.join(DATA_DIR, "user.json")
IMAGE_CHECKPOINT       = "image_feats_checkpoint.pt"
IMAGE_CHECKPOINT_INTVL = 100_000                         # checkpoint every 100k images
MAX_IMAGES_PER_USER    = 10                              # cap per user
IMAGE_BATCH_SIZE       = 512 
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"

# ─── SETUP ─────────────────────────────────────────────────────────────────
print(f"Using device: {DEVICE}")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                      .vision_model.to(DEVICE).eval()

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# state dictionaries
sum_img_feats   = {}   # user_id → running sum tensor [768]
img_counts      = {}   # user_id → how many images added
processed_images= 0

# ─── RESUME FROM CHECKPOINT ────────────────────────────────────────────────
if os.path.exists(IMAGE_CHECKPOINT):
    ckpt = torch.load(IMAGE_CHECKPOINT, map_location="cpu")
    # move back to DEVICE
    for uid, feat in ckpt['sum_img_feats'].items():
        sum_img_feats[uid] = feat.to(DEVICE)
    img_counts       = ckpt['img_counts']
    processed_images = ckpt['processed_images']
    print(f"▶ Resumed at {processed_images} images processed")

def save_checkpoint():
    """Dump partial sums/counts to disk so we can resume."""
    cpu_feats = { uid: feat.cpu()
                  for uid, feat in sum_img_feats.items() }
    torch.save({
        'sum_img_feats':   cpu_feats,
        'img_counts':      img_counts,
        'processed_images':processed_images
    }, IMAGE_CHECKPOINT)
    print(f"💾 Checkpoint @ {processed_images} images")

# ─── HELPERS ────────────────────────────────────────────────────────────────
def download_image(url):
    try:
        r = requests.get(url, timeout=5)
        return Image.open(BytesIO(r.content)).convert("RGB")
    except:
        return None

def embed_batch(image_list):
    """Returns a [N×768] tensor in half‐precision on DEVICE."""
    inputs = processor(images=image_list, return_tensors="pt").to(DEVICE)
    with torch.cuda.amp.autocast():
        out = model(**inputs)
    return out.pooler_output  # [N,768]

# ─── STREAM & EMBED ────────────────────────────────────────────────────────
img_buffer, uid_buffer = [], []

for fn in tqdm(sorted(glob.glob(os.path.join(DATA_DIR, "tweet_*.json"))),
               desc="Files"):
    with open(fn, 'r', encoding='utf-8') as f:
        for tw in tqdm(ijson.items(f, 'item'),
                       desc=os.path.basename(fn),
                       leave=False,
                       unit="imgs"):
            uid = tw.get('author_id')
            if img_counts.get(uid,0) >= MAX_IMAGES_PER_USER:
                continue

            # extract any media entries
            media = (
                tw.get('entities',{}).get('media',[]) +
                tw.get('extended_entities',{}).get('media',[])
            )
            # if media:
            #     print(f"Media: {media}")
            if not media:
                continue

            for m in media:
                url = m.get('media_url_https') or m.get('media_url') or m.get('url')
                if not url:
                    continue

                # 1) download
                try:
                    resp = requests.get(url, timeout=5)
                    img  = Image.open(BytesIO(resp.content)).convert("RGB")
                except Exception:
                    continue
                if img is None:
                    continue

                img_counts[uid] = img_counts.get(uid,0) + 1
                processed_images += 1

                # buffer for batch encode
                img_buffer.append(img)
                uid_buffer.append(uid)

                if len(img_buffer) >= IMAGE_BATCH_SIZE:
                    feats = embed_batch(img_buffer)  # [B,768]
                    for u, feat in zip(uid_buffer, feats):
                        if u not in sum_img_feats:
                            sum_img_feats[u] = torch.zeros(768, device=DEVICE).half()
                        sum_img_feats[u] += feat
                    img_buffer.clear()
                    uid_buffer.clear()

                # 4) checkpoint periodically
                if processed_images % IMAGE_CHECKPOINT_INTVL == 0:
                    save_checkpoint()

                # stop if this user hit its cap
                if img_counts[uid] >= MAX_IMAGES_PER_USER:
                    break

# final checkpoint
save_checkpoint()

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

# build one row-per-user
rows = []
for uid in ordered_uids:
    cnt = img_counts.get(uid, 0)
    if cnt > 0:
        rows.append(sum_img_feats[uid] / cnt)
    else:
        rows.append(torch.zeros(768, device=DEVICE))

image_feats = torch.stack(rows, dim=0)      # [num_users, 768]
torch.save(image_feats.cpu(), 'image_feats.pt')
print(f"✅ Saved image_feats.pt with shape {image_feats.shape}")
