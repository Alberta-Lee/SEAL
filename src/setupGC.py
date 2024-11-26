import random
from random import choices
import numpy as np
import pandas as pd
import scipy

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree

from models import GAT, GAT_dc, serverGAT, serverGAT_dc
from server import Server
from client import Client_GC
from utils import get_maxDegree, get_stats, split_data, get_numGraphLabels, init_structure_encoding

def _dirichletChunk(graphs, num_client, partition, alpha, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    graphs_chunks = []

    if partition == "iid":
        idxs = np.random.permutation(totalNum)
        batch_idxs = np.array_split(idxs, num_client)
        party2dataidx = {j: batch_idxs[j] for j in range(num_client)}
        for i in range(num_client):
            graphs_chunks.append([graphs[idx] for idx in party2dataidx[i]])
    elif partition == "non-iid":
        n_classes = get_numGraphLabels(graphs)
        label_distribution = np.random.dirichlet([alpha]*num_client, n_classes)
        labels = np.array([graph.y for graph in graphs]).reshape(-1)
        class_idxs = [np.argwhere(labels == i).flatten() for i in range(n_classes)]
        client_idxs = [[] for _ in range(num_client)]
        for k_idxs, fracs in zip(class_idxs, label_distribution):
            for i, idcs in enumerate(np.split(k_idxs, (np.cumsum(fracs)[:-1] * len(k_idxs)).astype(int))):

                while len(idcs) < 50:
                    idcs = np.concatenate((idcs, np.array([random.choice(k_idxs)])))
                client_idxs[i] += [idcs]
                
        client_idxs = [np.concatenate(idcs) for idcs in client_idxs]
        for i in range(num_client):
            graphs_chunks.append([graphs[idx] for idx in client_idxs[i]])
    return graphs_chunks


def prepareData_oneDS(args, datapath, data, num_client, batchSize, convert_x=False, seed=None):
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = [x for x in tudataset]
    print("  **", data, len(graphs))

    graphs_chunks = _dirichletChunk(graphs, num_client, args.partition, args.alpha, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)

        ds_train = init_structure_encoding(args, gs=ds_train, type_init=args.type_init)
        ds_val = init_structure_encoding(args, gs=ds_val, type_init=args.type_init)
        ds_test = init_structure_encoding(args, gs=ds_test, type_init=args.type_init)

        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)
    return splitedData, df


def prepareData_multiDS(args, datapath, group='chem', batchSize=32, seed=None):
    assert group in ['chem', "biochem", 'biochemsn', "biosncv"]

    if group == 'chem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    elif group == 'biochem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                               # bioinformatics
    elif group == 'biochemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",                               # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]                     # social networks
    elif group == 'biosncv':
        datasets = ["ENZYMES", "DD", "PROTEINS",                               # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI",                     # social networks
                    "Letter-high", "Letter-low", "Letter-med"]                 # computer vision

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
        graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
        graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    return splitedData, df


def js_diver(P,Q):
    M=P+Q
    return 0.5*scipy.stats.entropy(P,M,base=2)+0.5*scipy.stats.entropy(Q,M,base=2)


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if args.alg == 'fedstar':
            cmodel_gc = GAT_dc(num_node_features, args.n_se, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        else:
            cmodel_gc = GAT(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay, momentum=0.99)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))

    if args.alg == 'fedstar':
        smodel = serverGAT_dc(n_se=args.n_se, nlayer=args.nlayer, nhid=args.hidden)
    else:
        smodel = serverGAT(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients