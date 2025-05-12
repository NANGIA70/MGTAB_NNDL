#!/usr/bin/env python3
import pandas as pd, torch, json
from pathlib import Path
from tqdm import tqdm
import math

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR        = Path("/mnt/gcs/TwiBot-22")
EDGE_CSV        = DATA_DIR / "edge.csv"
LABEL_CSV       = DATA_DIR / "label.csv"
SPLIT_CSV       = DATA_DIR / "split.csv"
SAMPLE_UID_CSV  = "sample_uids_10k.csv"
OUT_EDGE_IDX    = "edge_index_10k.pt"
OUT_EDGE_TYPE   = "edge_type_10k.pt"
OUT_EDGE_WT     = "edge_weight_10k.pt"
OUT_LABEL       = "label_10k.pt"
CHUNK           = 20_000  # smaller for subset

REL_MAP = {
    "follower":   "followers",
    "friendship": "friends",
    "mention":    "mention",
    "reply":      "reply",
    "quote":      "quoted",
    "url":        "url",
    "hashtag":    "hashtag",
}
LABEL2IDX = {r: i for i, r in enumerate([
    "followers","friends","mention","reply","quoted","url","hashtag"])
}

# ─── 1) Load 10k sample uids ────────────────────────────────────────────────
sample = pd.read_csv(SAMPLE_UID_CSV, dtype=str)
ordered_uids = sample["id"].tolist()
uid2idx      = {uid: i for i, uid in enumerate(ordered_uids)}
uid_set      = set(ordered_uids)
print(f"→ {len(ordered_uids):,} sample users")

# ─── 2) Detect relation column the same way ────────────────────────────────
hdr = pd.read_csv(EDGE_CSV, nrows=1)
if   "relation_type" in hdr.columns: relcol="relation_type"
elif "relation"      in hdr.columns: relcol="relation"
elif "relationType"  in hdr.columns: relcol="relationType"
else: raise KeyError("No relation column found")
print(f"Relation column: '{relcol}'")

# ─── 3) Stream & filter edges ─────────────────────────────────────────────
SRC, DST, ET = [], [], []

num_lines = sum(1 for _ in open(EDGE_CSV, 'r', encoding='utf-8'))
num_chunks = math.ceil((num_lines - 1) / CHUNK)

for chunk in tqdm(
        pd.read_csv(EDGE_CSV, chunksize=CHUNK, dtype=str),
        desc="Edge‐chunks",
        total=num_chunks,
        unit="chunk",
):
    m = (
      chunk[relcol].isin(REL_MAP) &
      chunk["source_id"].isin(uid_set) &
      chunk["target_id"].isin(uid_set)
    )
    sub = chunk.loc[m, ["source_id","target_id",relcol]]
    if sub.empty: continue
    sub["rname"] = sub[relcol].map(REL_MAP)
    sub["tid"]   = sub["rname"].map(LABEL2IDX)
    sub["sidx"]  = sub["source_id"].map(uid2idx)
    sub["didx"]  = sub["target_id"].map(uid2idx)

    SRC.extend(sub["sidx"].tolist())
    DST.extend(sub["didx"].tolist())
    ET.extend(sub["tid"].tolist())

edge_index  = torch.stack([
    torch.tensor(SRC, dtype=torch.long),
    torch.tensor(DST, dtype=torch.long)
], dim=0)
edge_type   = torch.tensor(ET, dtype=torch.long)
edge_weight = torch.ones_like(edge_type, dtype=torch.float)

torch.save(edge_index, OUT_EDGE_IDX)
torch.save(edge_type,  OUT_EDGE_TYPE)
torch.save(edge_weight,OUT_EDGE_WT)
print("✔ Saved 10k edge files")

# ─── 4) Labels & masks for 10k ─────────────────────────────────────────────
lbl = pd.read_csv(LABEL_CSV, index_col="id", dtype={"id":str})
lbl["label_id"] = lbl["label"].map({"human":0,"bot":1})
y10k = torch.tensor(lbl.loc[ordered_uids,"label_id"].values, dtype=torch.long)
torch.save(y10k, OUT_LABEL)

spl = pd.read_csv(SPLIT_CSV, index_col="id", dtype={"id":str})
for sp in ["train","valid","test"]:
    mask = torch.tensor(
        (spl["split"]==sp).loc[ordered_uids].values,
        dtype=torch.bool
    )
    torch.save(mask, f"{sp}_mask_10k.pt")
    print(f"✔ Saved {sp}_mask_10k.pt")
