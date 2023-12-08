from dgl.data import CitationGraphDataset
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import torch_geometric
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS, NELL, WebKB, CoraFull
import os.path as osp
import torch_geometric.transforms as T

def download(dataset):
    if dataset == 'cora' or 'citeseer' or 'pubmed':
        return CitationGraphDataset(name=dataset)
    else:
        return None


def load(args):
    root = osp.join('~/public_data/pyg_data')
    transform = T.Compose([T.ToUndirected()])
    # Load a PyG dataset
    if args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, args.dataset, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
    elif args.dataset in {'Photo', 'Computers'}:
        dataset = Amazon(root, args.dataset, transform=T.NormalizeFeatures())
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
        dataset = CoraFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data", transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    dense_adjacency = torch_geometric.utils.to_dense_adj(data.edge_index)
    adj = dense_adjacency.numpy()[0]
    # Convert edge_index to a scipy sparse matrix
    import torch.nn.functional as F
    # labels = F.one_hot(data.y)
    # labels = labels.to(torch.long)
    labels = data.y
    feat = data.x.numpy()
    idx_train = torch.where(data.train_mask)[0]
    idx_val = torch.where(data.val_mask)[0]
    idx_test = torch.where(data.test_mask)[0]

    diff = compute_ppr(adj, 0.2)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, diff, feat, labels, idx_train, idx_val, idx_test



def load_forlink(args, run):
    root = osp.join('~/public_data/pyg_data')
    transform = T.Compose([T.ToUndirected()])
    # Load a PyG dataset
    if args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, args.dataset, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
    elif args.dataset in {'Photo', 'Computers'}:
        dataset = Amazon(root, args.dataset, transform=T.NormalizeFeatures())
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
        dataset = CoraFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data", transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                        is_undirected=True,
                                                        split_labels=True,
                                                        add_negative_train_samples=True)(data)
    torch.save(train_data, f'./data/{args.dataset}_traindata_{run}.pth')
    torch.save(val_data, f'./data/{args.dataset}_valdata_{run}.pth')
    torch.save(test_data, f'./data/{args.dataset}_testdata_{run}.pth')


    dense_adjacency = torch_geometric.utils.to_dense_adj(train_data.edge_index, max_num_nodes=data.x.shape[0])
    adj = dense_adjacency.numpy()[0]
    # Convert edge_index to a scipy sparse matrix
    import torch.nn.functional as F
    # labels = F.one_hot(data.y)
    # labels = labels.to(torch.long)
    labels = train_data.y
    feat = train_data.x.numpy()
    idx_train = torch.where(train_data.train_mask)[0]
    idx_val = torch.where(train_data.val_mask)[0]
    idx_test = torch.where(train_data.test_mask)[0]

    diff = compute_ppr(adj, 0.2)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, diff, feat, labels, idx_train, idx_val, idx_test

if __name__ == '__main__':
    load('cora')
