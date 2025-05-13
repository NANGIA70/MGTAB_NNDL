import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import time
import torch
from torch import nn
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Dataset import MGTABNew
from models import RGT_multimodal
from utils import sample_mask
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description=' RGT_multimodal')
parser.add_argument('--task', type=str, default='bot', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=int, default=[1], nargs='+', help='selection of random seeds')
parser.add_argument('--hidden_dimension', type=int, default=128, help='number of hidden units')
parser.add_argument("--out_channel", type=int, default=64, help="out channels")
parser.add_argument('--trans_head', type=int, default=4, help='number of trans_head')
parser.add_argument('--semantic_head', type=int, default=4, help='number of semantic_head')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer')
args = parser.parse_args()


def main(seed):

    args.num_edge_type = len(args.relation_select)

    # ─── DEVICE ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
   
    dataset = MGTABNew('./Dataset/TwiBot22-as-MGTAB-10k-new')
    data = dataset[0].to(device)

    args.out_dim = 2

    sample_number = len(data.y)
    args.features_num = data.x.shape[1]
    args.img_feat_dim = data.img.shape[1]

    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]


    data.train_mask = sample_mask(train_idx, sample_number)
    data.val_mask = sample_mask(val_idx, sample_number)
    data.test_mask = sample_mask(test_idx, sample_number)

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask
    data = data.to(device)

    model =  RGT_multimodal(args).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    index_select_list = (data.edge_type == 100)
    relation_dict = {
        0: 'followers',
        1: 'friends',
        2: 'mention',
        3: 'reply',
        4: 'quoted',
        5: 'url',
        6: 'hashtag'
    }

    print('relation used:', end=' ')
    for number, features_index in enumerate(args.relation_select):
        new_indx_select = (features_index == data.edge_type)
        index_select_list = index_select_list + new_indx_select
        data.edge_type[new_indx_select] = number
        print('{}'.format(relation_dict[features_index]), end='  ')
    edge_index = data.edge_index[:, index_select_list].long()
    edge_type = data.edge_type[index_select_list].long()
    edge_weight = data.edge_weight[index_select_list]


    def train(epoch):
        model.train()
        output = model(data.x, edge_index, edge_type, data.img)
        loss_train = loss(output[data.train_mask], data.y[data.train_mask])
        output = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_train = accuracy_score(label[train_mask], output[train_mask])
        acc_val = accuracy_score(label[val_mask], output[val_mask])
        optimizer.zero_grad()
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(loss_train.item())
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()), )
        return acc_val


    def test():
        model.eval()
        output = model(data.x, edge_index, edge_type, data.img)
        loss_test = loss(output[data.test_mask], data.y[data.test_mask])
        output = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_test = accuracy_score(label[test_mask], output[test_mask])
        f1 = f1_score(label[test_mask], output[test_mask], average='macro')
        precision = precision_score(label[test_mask], output[test_mask], average='macro')
        recall = recall_score(label[test_mask], output[test_mask], average='macro')
        return acc_test, loss_test, f1, precision, recall


    max_val_acc = 0
    total_loss = []
    for epoch in range(args.epochs):
        acc_val = train(epoch)
        acc_test, loss_test, f1, precision, recall = test()
        if acc_val > max_val_acc:
            max_val_acc = acc_val
            max_acc = acc_test
            max_epoch = epoch + 1
            max_f1 = f1
            max_precision = precision
            max_recall = recall

            print('Saving model...')
            torch.save(model.state_dict(), ' RGT_multimodal_{}_seed_{}.pkl'.format(args.task, seed))
            print('Model saved!')
    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "loss= {:.4f}".format(loss_test.item()),
          "test_accuracy= {:.4f}".format(max_acc),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1_score= {:.4f}".format(max_f1)
          )
    
    epochs = list(range(1, len(total_loss) + 1))
    plt.plot(epochs, total_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss vs Epoch')
    plt.tight_layout()
    plt.show()
    return max_acc, max_precision, max_recall, max_f1

if __name__ == "__main__":
    t = time.time()
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []
    for i, seed in enumerate(args.random_seed):
        print('traning {}th model\n'.format(i+1))
        acc, precision, recall, f1 = main(seed)
        acc_list.append(acc*100)
        precision_list.append(precision*100)
        recall_list.append(recall*100)
        f1_list.append(f1*100)

    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('total time:', time.time() - t)
