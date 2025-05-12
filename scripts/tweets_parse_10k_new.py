# tweets_parse_10k_new.py
#!/usr/bin/env python3
import os, glob, pickle, torch, ijson, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR            = "/mnt/gcs/TwiBot-22"
SAMPLE_UID_CSV      = "sample_uids_10k_new.csv"
TWEET_GLOB_PATTERN  = os.path.join(DATA_DIR, "tweet_*.json")
OUTPUT_FILE         = "tweets_tensor_10k_new.pt"
CHECKPOINT_FILE     = "tweets_feats_10k_new_ckpt.pkl"
CHECKPOINT_INTERVAL = 50_000
BATCH_SIZE          = 256
MAX_TWEETS_PER_USER = 20

# ─── SAMPLE UIDs ───────────────────────────────────────────────────────────
uids = pd.read_csv(SAMPLE_UID_CSV, dtype=str)['id'].tolist()
uid2idx = {u:i for i,u in enumerate(uids)}
num_users = len(uids)
print(f"Processing {num_users} users")

# ─── MODEL SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SentenceTransformer("LaBSE").to(device).eval()
if device.type=='cuda': torch.backends.cudnn.benchmark = True

# ─── PREALLOCATE ───────────────────────────────────────────────────────────
sum_embeds   = torch.zeros((num_users,768), device=device)
tweet_counts = torch.zeros(num_users, dtype=torch.long, device=device)

# ─── CHECKPOINT HELPERS ─────────────────────────────────────────────────────
def save_ckpt(n):
    torch.save({'sum_embeds': sum_embeds.cpu(),
                'tweet_counts':tweet_counts.cpu(),
                'n':n}, CHECKPOINT_FILE)
    print(f"Checkpoint @ {n} tweets")

processed = 0
if os.path.exists(CHECKPOINT_FILE):
    ck = torch.load(CHECKPOINT_FILE, map_location='cpu')
    sum_embeds.copy_(ck['sum_embeds'].to(device))
    tweet_counts.copy_(ck['tweet_counts'].to(device))
    processed = ck['n']
    print(f"Resumed @ {processed}")

# ─── FLUSH BATCH ───────────────────────────────────────────────────────────
batch_texts, batch_idxs = [], []
def flush():
    global batch_texts,batch_idxs,processed
    if not batch_texts: return
    embs = model.encode(batch_texts, convert_to_tensor=True,
                        batch_size=BATCH_SIZE, device=device, show_progress_bar=False)
    for idx, e in zip(batch_idxs, embs):
        sum_embeds[idx]   += e
        tweet_counts[idx] += 1
    processed += len(batch_texts)
    batch_texts, batch_idxs = [], []
    if processed % CHECKPOINT_INTERVAL < BATCH_SIZE:
        save_ckpt(processed)

# ─── STREAM TWEETS ─────────────────────────────────────────────────────────
for fn in tqdm(sorted(glob.glob(TWEET_GLOB_PATTERN)), desc="Files"):
    with open(fn,'r') as f:
        for tw in ijson.items(f,'item'):
            uid = tw.get('author_id')
            if uid not in uid2idx: continue
            idx = uid2idx[uid]
            if tweet_counts[idx] >= MAX_TWEETS_PER_USER: continue
            txt = tw.get('text','').strip()
            if not txt: continue
            batch_texts.append(txt); batch_idxs.append(idx)
            if len(batch_texts)>=BATCH_SIZE: flush()
# final
flush()
save_ckpt(processed)

# ─── AVERAGE & SAVE ────────────────────────────────────────────────────────
mask = tweet_counts>0
avg  = torch.zeros_like(sum_embeds)
avg[mask] = sum_embeds[mask]/tweet_counts[mask].unsqueeze(1)
torch.save(avg.cpu(), OUTPUT_FILE)
print(f"Saved tweets → {OUTPUT_FILE}  shape={avg.shape}")
