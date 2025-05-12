#!/usr/bin/env python3
import json, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
import math

# ─── CONFIG ────────────────────────────────────────────────────────────────
# DATA_DIR         = Path("/mnt/gcs/TwiBot-22")
DATA_DIR         = Path("../Dataset/TwiBot-22")
EDGE_CSV         = DATA_DIR / "edge.csv"
LABEL_CSV        = DATA_DIR / "label.csv"
SPLIT_CSV        = DATA_DIR / "split.csv"
USER_JSON        = DATA_DIR / "user.json"
OUTPUT_EDGE_IDX  = "edge_index.pt"
OUTPUT_EDGE_TYPE = "edge_type.pt"
OUTPUT_EDGE_WT   = "edge_weight.pt"
OUTPUT_LABEL     = "label.pt"
OUTPUT_TRAIN     = "train_mask.pt"
OUTPUT_VALID     = "valid_mask.pt"
OUTPUT_TEST      = "test_mask.pt"

# ─── 1) Load ordered_uids (unchanged) ─────────────────────────────────────
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
uid2idx = {uid: i for i, uid in enumerate(ordered_uids)}
uid_set = set(ordered_uids)
print(f"→ {len(ordered_uids):,} users loaded")

# ─── 2) Relation mapping ───────────────────────────────────────────────────
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

# ─── 3) Detect which column holds the relation name ────────────────────────
# We'll read one small chunk to see the header
sample = pd.read_csv(EDGE_CSV, nrows=1)
print(f"Detected {len(sample.columns):,} columns in {EDGE_CSV}: {', '.join(sample.columns)}")
if   "relation_type" in sample.columns: relcol = "relation_type"
elif "relation"      in sample.columns: relcol = "relation"
elif "relationType"  in sample.columns: relcol = "relationType"
else:
    raise KeyError(f"No relation column found in {EDGE_CSV} (tried relation_type, relation, relationType)")

print(f"Detected relation column: '{relcol}'")

# ─── 4) Stream edges in chunks & map ────────────────────────────────────────
SRC, DST, ET   = [], [], []

CHUNK = 10000  # adjust to memory
num_lines = sum(1 for _ in open(EDGE_CSV, 'r', encoding='utf-8'))
num_chunks = math.ceil((num_lines - 1) / CHUNK)


print("Processing edges in chunks…")
for chunk in tqdm(
        pd.read_csv(EDGE_CSV, chunksize=CHUNK, dtype=str),
        desc="Edge‐chunks",
        total=num_chunks,
        unit="chunk",
):
    mask = (
        chunk[relcol].isin(REL_MAP) &
        chunk["source_id"].isin(uid_set) &
        chunk["target_id"].isin(uid_set)
    )
    sub = chunk.loc[mask, ["source_id", "target_id", relcol]]
    if sub.empty:
        continue
    # map to MGTAB names and indices
    sub["rtype"] = sub[relcol].map(REL_MAP)
    sub["tid"]   = sub["rtype"].map(LABEL2IDX)
    sub["sidx"]  = sub["source_id"].map(uid2idx)
    sub["didx"]  = sub["target_id"].map(uid2idx)

    SRC.extend(sub["sidx"].tolist())
    DST.extend(sub["didx"].tolist())
    ET.extend(sub["tid"].tolist())

print(f"Total edges kept: {len(SRC):,}")

edge_index = torch.stack([
    torch.tensor(SRC, dtype=torch.long),
    torch.tensor(DST, dtype=torch.long)
], dim=0)
edge_type   = torch.tensor(ET, dtype=torch.long)
edge_weight = torch.ones(edge_type.size(0), dtype=torch.float)

torch.save(edge_index, OUTPUT_EDGE_IDX)
torch.save(edge_type,  OUTPUT_EDGE_TYPE)
torch.save(edge_weight,OUTPUT_EDGE_WT)
print("✔ Saved edge_index.pt, edge_type.pt, edge_weight.pt")

# ─── 5) Labels & splits ───────────────────────────────────────────────────
print("Processing labels & splits…")
lbl = pd.read_csv(LABEL_CSV, index_col="id", dtype={"id": str})
lbl["label_id"] = lbl["label"].map({"human": 0, "bot": 1})
y = torch.tensor(lbl.loc[ordered_uids, "label_id"].values, dtype=torch.long)
torch.save(y, OUTPUT_LABEL)

spl = pd.read_csv(SPLIT_CSV, index_col="id", dtype={"id": str})
for split in ["train", "valid", "test"]:
    m = torch.tensor(
        (spl["split"] == split).loc[ordered_uids].values,
        dtype=torch.bool
    )
    torch.save(m, {"train":"train","valid":"valid","test":"test"}[split] + "_mask.pt")
    print(f"✔ Saved {split}_mask.pt")
