# user_parse_10k_new.py
#!/usr/bin/env python3
import os, json, pandas as pd, torch, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR     = "/mnt/gcs/TwiBot-22"
SAMPLE_FILE  = "sample_uids_10k_new.csv"
OUTPUT_META  = "node_meta_10k_new.json"
OUTPUT_PROP  = "num_properties_tensor_10k_new.pt"
N_SAMPLE     = 5000  # per class

# ─── LOAD USERS & LABELS ───────────────────────────────────────────────────
DATA_DIR = Path(DATA_DIR)
def load_json_records(fname):
    path = DATA_DIR / fname
    with open(path, 'r', encoding='utf-8') as f:
        try:    data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
    return data

print("Loading users & labels…")
users = pd.DataFrame(load_json_records("user.json"))
labels= pd.read_csv(DATA_DIR/"label.csv", dtype=str).set_index("id")["label"]

# ─── BALANCED SAMPLE ───────────────────────────────────────────────────────
humans = labels[labels=="human"].sample(N_SAMPLE, random_state=42).index
bots   = labels[labels=="bot"].sample(N_SAMPLE, random_state=42).index
sample_uids = pd.Index(humans).append(bots)
sample_uids.to_series(name="id").to_csv(SAMPLE_FILE, index=False)
print(f"Saved balanced sample → {SAMPLE_FILE}")

# ─── FILTER TO SAMPLE ──────────────────────────────────────────────────────
df = users[users['id'].astype(str).isin(sample_uids)].reset_index(drop=True)

# ─── PREPROCESS META & PROPERTIES ──────────────────────────────────────────
pm  = pd.json_normalize(df['public_metrics'])
ent = pd.json_normalize(df['entities'])
df  = pd.concat([df.drop(['public_metrics','entities'],axis=1), pm, ent], axis=1)
df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
now = pd.Timestamp.now(tz="UTC")
df['account_age_days'] = (now - df['created_at']).dt.days
df['tweets_per_day']  = df['tweet_count'] / df['account_age_days']
df.drop(columns=['created_at'], errors='ignore', inplace=True)
df['profile_image_url'] = df['profile_image_url'].fillna('')
df[['protected','verified']] = df[['protected','verified']].astype(int)

# text combo + LaBSE embed
def safe_str(x): return x if isinstance(x,str) else ''
df['text_combo'] = (
    df['description'].apply(safe_str) + "  " +
    df['name'].apply(safe_str)        + "  " +
    df['username'].apply(safe_str)    + "  " +
    df['location'].apply(safe_str)
)
df.drop(columns=['description','name','username','location'], inplace=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SentenceTransformer("LaBSE").to(device).eval()
batch_size = 256
embs = []
for i in range(0, len(df), batch_size):
    batch = df['text_combo'].iloc[i:i+batch_size].tolist()
    em    = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embs.append(em)
import numpy as np
embeddings = np.vstack(embs)

# PCA → 12
pca = PCA(n_components=12, random_state=42)
text_feats = pca.fit_transform(embeddings)

# SAVE META
meta = {col: df[col].astype(str).tolist() for col in ['id','pinned_tweet_id','profile_image_url','url']}
with open(OUTPUT_META, 'w') as f: json.dump(meta, f)
print(f"Saved node_meta → {OUTPUT_META}")

# numeric + PCA → tensor
numeric_cols = ['protected','verified','followers_count','following_count',
                'tweet_count','listed_count','account_age_days','tweets_per_day']
num_feats = df[numeric_cols].to_numpy(dtype=float)
final    = np.hstack([num_feats, text_feats])
prop_tensor = torch.tensor(final, dtype=torch.float)
torch.save(prop_tensor, OUTPUT_PROP)
print(f"Saved properties → {OUTPUT_PROP}  shape={prop_tensor.shape}")
