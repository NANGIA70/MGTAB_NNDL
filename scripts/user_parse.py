import os, json, pandas as pd, torch
from collections import defaultdict
import tqdm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path

# ─── point at the same mountpoint ──────────────────────────
DATA_DIR     = "/mnt/gcs/TwiBot-22"
# USER_JSON    = os.path.join(DATA_DIR, "user.json")
# OUTPUT_DIR   = os.path.join(os.environ["HOME"], "MGTAB_NNDL")    # wherever you want to write your .pt files

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


# ─── Preprocessing ─────────────────────────────────
print("Preprocessing user data…")
pm = pd.json_normalize(users_df['public_metrics'])
users_df = pd.concat([users_df.drop('public_metrics',axis=1, errors="ignore"), pm], axis=1)

ent = pd.json_normalize(users_df['entities'])
users_df = pd.concat([users_df.drop('entities',axis=1, errors="ignore"), ent], axis=1)

# parse your created_at as UTC
users_df['created_at'] = pd.to_datetime(users_df['created_at'], utc=True)

# get “now” in UTC, so it’s also tz-aware
now_utc = pd.Timestamp.now(tz='UTC')

# 1) account age in days
users_df['account_age_days'] = (now_utc - users_df['created_at']).dt.days

# 2) tweets per day
users_df['tweets_per_day'] = users_df['tweet_count'] / users_df['account_age_days']

# Drop created at, description, and verified
users_df = users_df.drop(columns=['created_at'], errors="ignore")

users_df['profile_image_url'] = users_df['profile_image_url'].fillna('')

users_df['protected'] = users_df['protected'].astype(int)
users_df['verified']  = users_df['verified'].astype(int)

# ─── Text ─────────────────────────────────
print("Concatenating text fields and embedding…")
# combine the three text fields into one string per user
def safe_str(x):
    return x if isinstance(x, str) else ""
users_df['text_combo'] = (
    users_df['description'].apply(safe_str) + "  " +
    users_df['name'].apply(safe_str) + "  " +
    users_df['username'].apply(safe_str) + "  " +
    users_df['location'].apply(safe_str)
)

users_df['text_combo'] = users_df['text_combo'].fillna('')
users_df = users_df.drop(columns=['description', 'name', 'username', 'location'], errors="ignore")

users_df = users_df.drop(columns=['withheld', 'url.urls', 'description.urls', 'description.mentions', 'description.hashtags', 'description.cashtags'], errors="ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("LaBSE").to(device).eval()

batch_size = 256
texts = users_df['text_combo'].tolist()
embeddings = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embeddings.append(embs)
embeddings = np.vstack(embeddings)  


# -----------------------------------------------------------------------------
# ——— STEP 4: PCA → 8 DIMS ————————————————————————————————
# -----------------------------------------------------------------------------
print("[4/6] Reducing embeddings to 12 dims with PCA…")
pca = PCA(n_components=12, random_state=42)
text_feats_12 = pca.fit_transform(embeddings)  


id_cols = ['id', 'pinned_tweet_id', 'profile_image_url', 'url']
# ensure strings and fill missing pinned_tweet_id with empty string
users_df['pinned_tweet_id'] = users_df['pinned_tweet_id'].fillna('').astype(str)
users_df['profile_image_url'] = users_df['profile_image_url'].fillna('').astype(str)
users_df['url']               = users_df['url'].fillna('').astype(str)
users_df['id']                = users_df['id'].astype(str)

meta = { col: users_df[col].tolist() for col in id_cols }
with open('node_meta.json','w') as f:
    json.dump(meta, f)
print(f"[✔] Saved ID/meta mapping to node_meta.json")


numeric_cols = [
    'protected','verified',
    'followers_count','following_count','tweet_count','listed_count',
    'account_age_days','tweets_per_day'
]
numeric_feats = users_df[numeric_cols].to_numpy(dtype=float)
assert numeric_feats.shape[1] == 8


# -----------------------------------------------------------------------------
# ——— STEP 5: STACK & FORM FINAL (num_users × 20) ————————————————————
# -----------------------------------------------------------------------------
print("[5/6] Stacking numeric + text feats → (N, 20)…")
final_feats = np.hstack([numeric_feats, text_feats_12])
assert final_feats.shape[1] == 20, "Expected 20 features!"

# convert → torch tensor
prop_tensor = torch.tensor(final_feats, dtype=torch.float)

# -----------------------------------------------------------------------------
# ——— STEP 6: SAVE FOR RGT ————————————————————————————————
# -----------------------------------------------------------------------------
out_path = 'num_properties_tensor.pt'
torch.save(prop_tensor, out_path)
print(f"[6/6] Done! Saved property tensor with shape {prop_tensor.shape} to {out_path}")
