#!/usr/bin/env python3
import os, glob, pickle, torch, ijson
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR               = "/mnt/gcs/TwiBot-22"
SAMPLE_UID_CSV         = "sample_uids_10k.csv"
TWEET_GLOB_PATTERN     = os.path.join(DATA_DIR, "tweet_*.json")
OUTPUT_FILE            = "tweets_tensor_10k.pt"
CHECKPOINT_FILE        = "tweets_feats_10k_checkpoint.pkl"
CHECKPOINT_INTERVAL    = 50_000   # checkpoint every 50k tweets
BATCH_SIZE             = 256
MAX_TWEETS_PER_USER    = 20

# ─── LOAD SAMPLE UIDs ─────────────────────────────────────────────────────
sample_df = pd.read_csv(SAMPLE_UID_CSV, dtype=str)
sample_uids = sample_df['id'].tolist()
uid2idx = {uid: idx for idx, uid in enumerate(sample_uids)}
num_users = len(sample_uids)
print(f"Processing {num_users} sampled users from {SAMPLE_UID_CSV}")

# ─── MODEL SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SentenceTransformer("LaBSE").to(device).eval()
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# ─── PREALLOCATE STORAGE ───────────────────────────────────────────────────
sum_embeds   = torch.zeros((num_users, 768), device=device)
tweet_counts = torch.zeros(num_users, dtype=torch.long, device=device)

# ─── CHECKPOINT HELPERS ─────────────────────────────────────────────────────
def save_checkpoint(processed):
    torch.save({
        'sum_embeds':   sum_embeds.cpu(),
        'tweet_counts': tweet_counts.cpu(),
        'processed':    processed
    }, CHECKPOINT_FILE)
    print(f"💾 Checkpoint saved at {processed} tweets")

# resume if exists
tweets_processed = 0
if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE, map_location='cpu')
    sum_embeds.copy_(ckpt['sum_embeds'].to(device))
    tweet_counts.copy_(ckpt['tweet_counts'].to(device))
    tweets_processed = ckpt['processed']
    print(f"Resumed from checkpoint: {tweets_processed} tweets processed")

# ─── BATCH BUFFER ───────────────────────────────────────────────────────────
batch_texts = []
batch_idxs  = []

def flush_batch():
    global batch_texts, batch_idxs, tweets_processed
    if not batch_texts:
        return
    # encode batch
    embs = model.encode(
        batch_texts,
        convert_to_tensor=True,
        batch_size=BATCH_SIZE,
        device=device,
        show_progress_bar=False
    )  # [B,768]
    # accumulate
    for idx, emb in zip(batch_idxs, embs):
        sum_embeds[idx]   += emb
        tweet_counts[idx] += 1
    tweets_processed += len(batch_texts)
    batch_texts, batch_idxs = [], []
    # checkpoint
    if tweets_processed % CHECKPOINT_INTERVAL < BATCH_SIZE:
        save_checkpoint(tweets_processed)

# ─── STREAM & PROCESS ──────────────────────────────────────────────────────
for fn in tqdm(sorted(glob.glob(TWEET_GLOB_PATTERN)), desc="Files"):
    with open(fn, 'r', encoding='utf-8') as f:
        for tw in tqdm(ijson.items(f, 'item'),
                       desc=f"Parsing {os.path.basename(fn)}",
                       leave=False,
                       unit="tweets"):
            uid = tw.get('author_id')
            if uid not in uid2idx:
                continue
            idx = uid2idx[uid]
            # cap tweets per user
            if tweet_counts[idx] >= MAX_TWEETS_PER_USER:
                continue
            text = tw.get('text', '').strip()
            if not text:
                continue
            batch_texts.append(text)
            batch_idxs.append(idx)
            if len(batch_texts) >= BATCH_SIZE:
                flush_batch()
# final flush
flush_batch()
# final checkpoint
save_checkpoint(tweets_processed)

# ─── AVERAGE & SAVE ─────────────────────────────────────────────────────────
print("Averaging embeddings and saving final tensor…")
mask = tweet_counts > 0
avg_embeds = torch.zeros_like(sum_embeds)
avg_embeds[mask] = sum_embeds[mask] / tweet_counts[mask].unsqueeze(1)
# save to disk
torch.save(avg_embeds.cpu(), OUTPUT_FILE)
print(f"✅ Saved tweet features to {OUTPUT_FILE} (shape {avg_embeds.size()})")
