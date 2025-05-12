#!/usr/bin/env python3
import os, json, pandas as pd, torch, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR     = "/mnt/gcs/TwiBot-22"   # directory containing user.json
SAMPLE_FILE  = "sample_uids_10k.csv"
OUTPUT_META  = "node_meta_10k.json"
OUTPUT_PROP  = "num_properties_tensor_10k.pt"
N_SAMPLE     = 10000

# ─── LOAD USERS ─────────────────────────────────────────────────────────────
DATA_DIR = Path(DATA_DIR)
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

def load_json_records(fname):
    path = DATA_DIR / fname
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
    return data

print("Loading user data…")
users = load_json_records('user.json')
print(f"Loaded {len(users):,} users. Sampling {N_SAMPLE}…")
users_df = pd.DataFrame(users)

# ─── SAMPLE 10k IDs ─────────────────────────────────────────────────────────
sample_df = users_df.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)
sample_uids = sample_df['id'].astype(str).tolist()
pd.Series(sample_uids, name='id').to_csv(SAMPLE_FILE, index=False)
print(f"Saved sample IDs to {SAMPLE_FILE}")

# ─── KEEP ONLY SAMPLE ───────────────────────────────────────────────────────
users_df = sample_df

# ─── PREPROCESS METRICS ─────────────────────────────────────────────────────
print("Preprocessing user data…")
pm = pd.json_normalize(users_df['public_metrics'])
users_df = pd.concat([users_df.drop('public_metrics',axis=1), pm], axis=1)
ent = pd.json_normalize(users_df['entities'])
users_df = pd.concat([users_df.drop('entities',axis=1), ent], axis=1)
users_df['created_at'] = pd.to_datetime(users_df['created_at'], utc=True)
now_utc = pd.Timestamp.now(tz='UTC')
users_df['account_age_days'] = (now_utc - users_df['created_at']).dt.days
users_df['tweets_per_day'] = users_df['tweet_count'] / users_df['account_age_days']
users_df.drop(columns=['created_at'], errors='ignore', inplace=True)

users_df['profile_image_url'] = users_df['profile_image_url'].fillna('')
users_df['protected'] = users_df['protected'].astype(int)
users_df['verified']  = users_df['verified'].astype(int)

# ─── TEXT COMBO & EMBED ────────────────────────────────────────────────────
print("Concatenating text fields and embedding…")
def safe_str(x): return x if isinstance(x, str) else ''
users_df['text_combo'] = (
    users_df['description'].apply(safe_str) + '  ' +
    users_df['name'].apply(safe_str) + '  ' +
    users_df['username'].apply(safe_str) + '  ' +
    users_df['location'].apply(safe_str)
)
users_df.drop(columns=['description','name','username','location'], inplace=True, errors='ignore')

# ─── MODEL SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SentenceTransformer("LaBSE").to(device).eval()
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    
batch_size = 256
embs = []
for i in range(0, len(users_df), batch_size):
    batch = users_df['text_combo'].iloc[i:i+batch_size].tolist()
    em = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embs.append(em)
embeddings = np.vstack(embs)

# ─── PCA → 12 dims ──────────────────────────────────────────────────────────
print("Reducing embeddings to 12 dims with PCA…")
pca = PCA(n_components=12, random_state=42)
text_feats_12 = pca.fit_transform(embeddings)

# ─── SAVE META JSON ─────────────────────────────────────────────────────────
id_cols = ['id','pinned_tweet_id','profile_image_url','url']
meta = {col: users_df[col].fillna('').astype(str).tolist() for col in id_cols}
with open(OUTPUT_META, 'w') as f:
    json.dump(meta, f)
print(f"Saved ID/meta mapping to {OUTPUT_META}")

# ─── STACK & SAVE PROPERTIES ─────────────────────────────────────────────────
numeric_cols = ['protected','verified','followers_count','following_count','tweet_count','listed_count','account_age_days','tweets_per_day']
numeric_feats = users_df[numeric_cols].to_numpy(dtype=float)
final_feats = np.hstack([numeric_feats, text_feats_12])  # shape (10k,20)
prop_tensor = torch.tensor(final_feats, dtype=torch.float)
torch.save(prop_tensor, OUTPUT_PROP)
print(f"Saved property tensor -> {OUTPUT_PROP}, shape={prop_tensor.shape}")
