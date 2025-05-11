import os, glob, pickle, torch, ijson
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
import json

# ─── NEW: point at the mountpoint ──────────────────────────────
DATA_DIR = "/mnt/gcs/TwiBot-22"      # <-- GCS is now mounted here
CHECKPOINT_FILE = "tweet_feats_checkpoint.pkl"
CHECKPOINT_INTERVAL  = 1_000_000        # save every 1M tweets
BATCH_SIZE      = 128

tweet_files = sorted(glob.glob(os.path.join(DATA_DIR, "tweet_*.json")))
sum_embeds    = defaultdict(lambda: torch.zeros(768))
tweet_counts  = defaultdict(int)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("LaBSE").to(device).eval()  # or 'cpu'
processed     = 0

# (optional) resume checkpoint
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE,'rb') as f:
        data = pickle.load(f)
        sum_embeds, tweet_counts, processed = data.values()
        print(f"Resumed at {processed} tweets")

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

batch_uids, batch_texts = [], []
def flush_batch():
    """Encode batch_texts, accumulate into sum_embeds/tweet_counts,
       advance processed counter, and checkpoint if needed."""
    global processed
    if not batch_texts:
        return

    # 2a) Batch-encode
    embs = model.encode(
        batch_texts,
        convert_to_tensor=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )
    # 2b) Accumulate
    for uid, emb in zip(batch_uids, embs):
        sum_embeds[uid]   += emb
        tweet_counts[uid] += 1

    # 2c) Update processed count & clear buffers
    processed += len(batch_texts)
    batch_uids.clear()
    batch_texts.clear()

    # 2d) Checkpoint?
    if processed and processed % CHECKPOINT_INTERVAL < BATCH_SIZE:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump({
                'sum_embeds':   sum_embeds,
                'tweet_counts': tweet_counts,
                'processed':    processed
            }, f)
        print(f"Checkpoint saved at {processed} tweets.")

# ─── 3) Stream & process with tqdm ───────────────────────────────────────────
for fn in tweet_files:
    with open(fn, 'r') as f:
        # ijson.items streams each JSON object in the top-level array
        for tw in tqdm(ijson.items(f, 'item'),
                       desc=f"Streaming {os.path.basename(fn)}",
                       leave=False):
            text = tw.get('text','').strip()
            if not text:
                continue

            batch_uids.append(tw['author_id'])
            batch_texts.append(text)

            if len(batch_texts) >= BATCH_SIZE:
                flush_batch()

flush_batch()

with open(CHECKPOINT_FILE, 'wb') as f:
    pickle.dump({
        'sum_embeds':   sum_embeds,
        'tweet_counts': tweet_counts,
        'processed':    processed
    }, f)
print(f"✅ Done! Total tweets processed: {processed}")

user_tweet_feats = []
for uid in ordered_uids:
    cnt = tweet_counts.get(uid, 0)
    if cnt > 0:
        avg = sum_embeds[uid] / cnt
    else:
        avg = torch.zeros(768)     # no tweets → zero vector
    user_tweet_feats.append(avg)

# shape [num_users, 768]
tweets_tensor = torch.stack(user_tweet_feats, dim=0)

# sanity check
print("Tweet‐feature tensor size:", tweets_tensor.shape)
# should be (len(ordered_uids), 768)

torch.save(tweets_tensor, 'tweets_tensor.pt')


