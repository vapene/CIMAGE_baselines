import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process
import warnings
warnings.filterwarnings("ignore")

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
# lr = 0.001
# l2_coef = 0.0
# hid_units = 512
sparse = True

lr = 0.001
l2_coef=5e-3
hid_units =128

nonlinearity = 'prelu'  # special name to separate parameters
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS, NELL, WebKB, CoraFull
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--device", type=int, default=0)

args = parser.parse_args()
print('args', args)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

root = osp.join('~/public_data/pyg_data')
transform = T.Compose([T.ToUndirected()])
# Load a PyG dataset
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
else:
    raise ValueError(args.dataset)

for run in range(10):
    target_shape = data.x.shape[0]
    while True:
        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                                    is_undirected=True,
                                                                    split_labels=True,
                                                                    add_negative_train_samples=True)(data)
        args.input_dim = data.x.shape[1]
        # Convert edge_index to a scipy sparse matrix
        adj = to_scipy_sparse_matrix(train_data.edge_index)
        if adj.shape[0]==target_shape:
            break
    torch.save(train_data, f'./data/{args.dataset}_traindata_{run}.pth')
    torch.save(val_data, f'./data/{args.dataset}_valdata_{run}.pth')
    torch.save(test_data, f'./data/{args.dataset}_testdata_{run}.pth')
    import torch.nn.functional as F

    labels = F.one_hot(data.y)  # Convert the feature matrix to a sparse matrix\
    labels = labels.float()
    features = sp.csr_matrix(data.x.numpy())
    idx_train = torch.where(data.train_mask)[0]
    idx_val = torch.where(data.val_mask)[0]
    idx_test = torch.where(data.test_mask)[0]

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = DGI(ft_size, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.to(device)
        features = features.to(device)
        if sparse:
            sp_adj = sp_adj.to(device)
        else:
            adj = adj.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.to(device)
            lbl = lbl.to(device)

        logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), f'./data/{args.dataset}_best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(f'./data/{args.dataset}_best_dgi.pkl'))

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    torch.save(embeds[0], f'./data/{args.dataset}_{run}.pt')


