import os.path as osp
import time
import argparse
import random
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class DotEdgeDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)
        return x.sigmoid()


def link_test(z, pos_edge_index, neg_edge_index):
    pos_pred = edge_decoder(z, pos_edge_index).squeeze().cpu()
    neg_pred = edge_decoder(z, neg_edge_index).squeeze().cpu()

    pred = torch.cat([pos_pred, neg_pred], dim=0)
    pos_y = pos_pred.new_ones(pos_pred.size(0))
    neg_y = neg_pred.new_zeros(neg_pred.size(0))

    y = torch.cat([pos_y, neg_y], dim=0)
    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred)



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

AUC_list=[]
AP_list=[]

for i in range(10):
    set_seed(i)
    # train_split Data(x=[2708, 1433], edge_index=[2, 8976], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], pos_edge_label=[4488], pos_edge_label_index=[2, 4488])
    train_split = torch.load(f"./data/{args.dataset}_traindata_{i}.pth").to(device)
    print('64')
    valid_split = torch.load(f"./data/{args.dataset}_valdata_{i}.pth").to(device)
    test_split = torch.load(f"./data/{args.dataset}_testdata_{i}.pth").to(device)
    rep = torch.load(f"./data/{args.dataset}_{i}.pt").to(device)
    print('70')
    splits = dict(train=train_split, valid=valid_split, test=test_split)
    edge_decoder = DotEdgeDecoder().to(device)

    valid_auc, valid_ap = link_test(rep, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index)
    test_auc, test_ap = link_test(rep, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index)
    print(f"i {i} test_auc {test_auc} test_ap {test_ap} ")
    AUC_list.append(test_auc)
    AP_list.append(test_ap)
print('auc_list',AUC_list)
print('ap_list',AP_list)
print('AUC',np.mean(AUC_list), np.std(AUC_list))
print('AP',np.mean(AP_list), np.std(AP_list))


file_name = f"./link_results/{args.dataset.lower()}.txt"
with open(file_name, 'w') as file:
    file.write(f"\n {args.dataset.lower()}"
               f"\n AUC_test: {AUC_list}, AP_test: {AP_list}"
               f"\n{np.mean(AUC_list) * 100:.2f}±{np.std(AUC_list)*100:.2f}"
               f"\n{np.mean(AP_list) * 100:.2f}±{np.std(AP_list)*100:.2f}")



