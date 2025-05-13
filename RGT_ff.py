import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Dataset import MGTABNew
from models import RGT_multimodal_feedforward
from utils import sample_mask
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='RGT_multimodal_feedforward')
parser.add_argument('--task', type=str, default='bot', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1,2,3,4,5,6], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=int, default=[1], nargs='+', help='selection of random seeds')
parser.add_argument('--hidden_dimension', type=int, default=128, help='number of hidden units')
parser.add_argument("--out_channel", type=int, default=64, help="out channels")
parser.add_argument('--trans_head', type=int, default=4, help='number of trans_head')
parser.add_argument('--semantic_head', type=int, default=4, help='number of semantic_head')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer')
parser.add_argument('--threshold', type=float, default=0.5, help='decision threshold for positive class')
args = parser.parse_args()

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(logits, targets, reduction='none')
    p_t = torch.exp(-ce)
    loss = alpha * (1 - p_t)**gamma * ce
    return loss.mean()

def main(seed):
    args.num_edge_type = len(args.relation_select)

    # ─── DEVICE ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ─── DATA ───────────────────────────────────────────────────────────────────
    dataset = MGTABNew('./Dataset/TwiBot22-as-MGTAB-10k-new')
    data = dataset[0].to(device)

    # print class balance
    counts = torch.bincount(data.y, minlength=2)
    print("Counts of each class:", counts)

    args.out_dim      = 2
    args.features_num = data.x.shape[1]
    args.img_feat_dim = data.img.shape[1]

    # create train/val/test splits
    N = data.y.size(0)
    idx = shuffle(np.arange(N), random_state=seed)
    train_end = int(0.7 * N)
    val_end   = int(0.9 * N)
    train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    data.train_mask = sample_mask(train_idx, N)
    data.val_mask   = sample_mask(val_idx, N)
    data.test_mask  = sample_mask(test_idx, N)

    # ─── MODEL, LOSS, OPTIM ────────────────────────────────────────────────────
    model     = RGT_multimodal_feedforward(args).to(device)

    pos_weight = torch.tensor([1.0, 1.5], device=device) 
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    # select edge types
    index_select = data.edge_type == 100
    relation_dict = {0:'followers',1:'friends',2:'mention',3:'reply',4:'quoted',5:'url',6:'hashtag'}
    print('relation used:', end=' ')
    for new_id, rel in enumerate(args.relation_select):
        mask = data.edge_type == rel
        index_select |= mask
        data.edge_type[mask] = new_id
        print(f"{relation_dict[rel]}", end='  ')
    print()

    edge_index  = data.edge_index[:, index_select].long()
    edge_type   = data.edge_type[index_select].long()
    edge_weight = data.edge_weight[index_select]

    total_loss = []

    def train(epoch):
        model.train()
        logits = model(data.x, edge_index, edge_type, data.img)
        loss_train = criterion(logits[data.train_mask], data.y[data.train_mask])

        # compute train/val accuracy with argmax
        preds_train = logits.max(1)[1]
        y = data.y
        acc_train = accuracy_score(y[data.train_mask].cpu().numpy(),
                                   preds_train[data.train_mask].cpu().numpy())
        acc_val   = accuracy_score(y[data.val_mask].cpu().numpy(),
                                   preds_train[data.val_mask].cpu().numpy())

        optimizer.zero_grad()
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss.append(loss_train.item())
        print(f"Epoch {epoch+1:03d} | train_loss: {loss_train:.4f} | "
              f"acc_train: {acc_train:.4f} | acc_val: {acc_val:.4f}")
        return acc_val

    def test():
        model.eval()
        logits = model(data.x, edge_index, edge_type, data.img)

        # 1) loss
        loss_test = criterion(logits[data.test_mask], data.y[data.test_mask])

        # 2) get positive-class probabilities & threshold
        probs     = F.softmax(logits, dim=1)
        pos_probs = probs[:, 1]
        preds     = (pos_probs > args.threshold).long()

        # 3) slice & move to CPU numpy
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = preds[data.test_mask].cpu().numpy()

        # 4) metrics
        acc_t  = accuracy_score(y_true, y_pred)
        prec   = precision_score(y_true, y_pred)
        rec    = recall_score(y_true, y_pred)
        f1     = f1_score(y_true, y_pred)

        return acc_t, loss_test.item(), prec, rec, f1

    # ─── RUN ────────────────────────────────────────────────────────────────────
    max_val_acc = 0
    best = None
    for epoch in range(args.epochs):
        val_acc = train(epoch)
        test_acc, test_loss, test_prec, test_rec, test_f1 = test()
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best = (epoch+1, test_acc, test_loss, test_prec, test_rec, test_f1)
            print("→ New best! Saving model...")
            torch.save(model.state_dict(), f"RGT_ff_{args.task}_seed{seed}.pth")

    # final results
    epoch_, acc_, loss_, prec_, rec_, f1_ = best
    print(f"\nTest set results @ epoch {epoch_}:")
    print(f"  loss:      {loss_:.4f}")
    print(f"  accuracy:  {acc_:.4f}")
    print(f"  precision: {prec_:.4f}")
    print(f"  recall:    {rec_:.4f}")
    print(f"  F1 score:  {f1_:.4f}")

    # plot
    plt.plot(range(1, len(total_loss)+1), total_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss vs Epoch')
    plt.tight_layout()
    plt.show()

    return acc_, prec_, rec_, f1_


if __name__ == "__main__":
    start = time.time()
    accs, precs, recs, f1s = [], [], [], []
    for i, seed in enumerate(args.random_seed, start=1):
        print(f"\n=== Training run #{i} (seed={seed}) ===")
        a, p, r, f = main(seed)
        accs.append(a*100)
        precs.append(p*100)
        recs.append(r*100)
        f1s.append(f*100)

    print("\nAcross runs:")
    print(f"acc:       {np.mean(accs):.2f} ± {np.std(accs):.2f}")
    print(f"precision: {np.mean(precs):.2f} ± {np.std(precs):.2f}")
    print(f"recall:    {np.mean(recs):.2f} ± {np.std(recs):.2f}")
    print(f"f1:        {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")
    print("total time:", time.time() - start)