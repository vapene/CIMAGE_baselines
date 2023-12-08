import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS, NELL, WebKB, CoraFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
import warnings
warnings.filterwarnings("ignore")

def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, final=False):
    x = data.x
    edge_index = data.edge_index
    y = data.y
    model.eval()
    z = model(x, edge_index)
    val_acc, test_acc =label_classification(z, y, data.train_mask.detach().cpu(),data.val_mask.detach().cpu(),data.test_mask.detach().cpu() )
    return val_acc, test_acc, z


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    print('args', args)
    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(root, name):

        transform = T.Compose([T.ToUndirected()])
        if args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
            dataset = Planetoid(root, args.dataset, transform=T.NormalizeFeatures())
            data = transform(dataset[0])
        elif args.dataset in {'Photo', 'Computers'}:
            dataset = Amazon(root, args.dataset)
            data = transform(dataset[0])
            data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        elif args.dataset in {'CS', 'Physics'}:
            dataset = Coauthor(root, args.dataset)
            data = transform(dataset[0])
            data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        elif args.dataset in {'WikiCS'}:
            dataset = WikiCS(root=root)
            data = transform(dataset[0])
            data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        elif args.dataset in {'CoraFull'}:
            dataset = CoraFull(root="./gae/data", transform=T.NormalizeFeatures())
            data = transform(dataset[0])
            data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        return data

    root = osp.join('~/public_data/pyg_data')
    data = get_dataset(root, args.dataset)

    acc_list = []
    for run in range(10):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        encoder = Encoder(data.num_features, num_hidden, activation,
                          base_model=base_model, k=num_layers).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        start = t()
        prev = start
        print("====== node start ======")
        best_val=0
        best_test = 0
        cnt=0
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, data.edge_index)

            val_acc, test_acc, z = test(model, data, final=True)
            if best_val < val_acc:
                best_val = val_acc
                print(f"best {epoch} val {val_acc}")
                best_test = test_acc
                cnt=0
            else:
                cnt+=1
            if cnt ==20:
                break

        acc_list.append(best_test)

    file_name = f"./node_results/{args.dataset.lower()}.txt"
    with open(file_name, 'w') as file:
        file.write(f"\n {args.dataset.lower()}"
                   f"\n accs: {acc_list*100}"
                   f"\n{np.mean(acc_list)*100:.2f}±{np.std(acc_list)*100:.2f}")