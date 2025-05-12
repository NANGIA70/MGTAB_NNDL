# edge_labels_10k_new.py
#!/usr/bin/env python3
import json, pandas as pd, torch
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR        = Path("/mnt/gcs/TwiBot-22")
EDGE_CSV        = DATA_DIR/"edge.csv"
LABEL_CSV       = DATA_DIR/"label.csv"
SPLIT_CSV       = DATA_DIR/"split.csv"
SAMPLE_UID_CSV  = "sample_uids_10k_new.csv"
OUT_EDGE_IDX    = "edge_index_10k_new.pt"
OUT_EDGE_TYPE   = "edge_type_10k_new.pt"
OUT_EDGE_WT     = "edge_weight_10k_new.pt"
OUT_LABEL       = "label_10k_new.pt"
CHUNK           = 200_000

# ─── SAMPLE UIDs ───────────────────────────────────────────────────────────
uids = pd.read_csv(SAMPLE_UID_CSV,dtype=str)['id'].tolist()
uid2idx = {u:i for i,u in enumerate(uids)}
uid_set = set(uids)
print(f"Bal‐sample: {len(uids)} users")

# ─── REL‐MAP ────────────────────────────────────────────────────────────────
REL_MAP = {'follower':'followers','friendship':'friends','mention':'mention',
           'reply':'reply','quote':'quoted','url':'url','hashtag':'hashtag'}
LABEL2IDX = {r:i for i,r in enumerate(['followers','friends','mention','reply','quoted','url','hashtag'])}

# ─── DETECT REL COL ─────────────────────────────────────────────────────────
hdr = pd.read_csv(EDGE_CSV, nrows=1)
for c in ('relation_type','relation','relationType'):
    if c in hdr.columns:
        relcol=c; break
else:
    raise KeyError("No relation column")

print("Using relcol:",relcol)

# ─── STREAM & FILTER ────────────────────────────────────────────────────────
SRC,DST,ET = [],[],[]
for chunk in pd.read_csv(EDGE_CSV, chunksize=CHUNK, dtype=str):
    m = (chunk[relcol].isin(REL_MAP) &
         chunk['source_id'].isin(uid_set) &
         chunk['target_id'].isin(uid_set))
    sub = chunk.loc[m, ['source_id','target_id',relcol]]
    if sub.empty: continue
    sub['rname']=sub[relcol].map(REL_MAP)
    sub['tid']  = sub['rname'].map(LABEL2IDX)
    sub['sidx'] = sub['source_id'].map(uid2idx)
    sub['didx'] = sub['target_id'].map(uid2idx)
    SRC.extend(sub['sidx'])
    DST.extend(sub['didx'])
    ET.extend(sub['tid'])

edge_index  = torch.stack([torch.tensor(SRC), torch.tensor(DST)], dim=0)
edge_type   = torch.tensor(ET, dtype=torch.long)
edge_weight = torch.ones_like(edge_type, dtype=torch.float)
torch.save(edge_index, OUT_EDGE_IDX)
torch.save(edge_type,  OUT_EDGE_TYPE)
torch.save(edge_weight,OUT_EDGE_WT)
print("Saved edges → *_new.pt")

# ─── LABELS & SPLITS ───────────────────────────────────────────────────────
lbl = pd.read_csv(LABEL_CSV,index_col="id",dtype=str)
lbl['label_id']=lbl['label'].map({'human':0,'bot':1})
y10k = torch.tensor(lbl.loc[uids,'label_id'].values, dtype=torch.long)
torch.save(y10k, OUT_LABEL)

spl = pd.read_csv(SPLIT_CSV,index_col="id",dtype=str)
for sp in ('train','valid','test'):
    m = (spl['split']==sp).loc[uids].values
    mask = torch.tensor(m, dtype=torch.bool)
    torch.save(mask, f"{sp}_mask_10k_new.pt")
    print(f"Saved {sp}_mask_10k_new.pt")
