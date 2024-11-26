import os, sys
import argparse
import random
import copy
import numpy as np

import torch
from tensorboardX import SummaryWriter
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "src").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setupGC
from training import *


def process_seal(args, clients, server, summary_writer):
    print("\nDone setting up SEAL devices.")

    print("Running SEAL ...")
    frame = run_seal(args, clients, server, args.num_rounds, args.local_epoch, samp=None, summary_writer=summary_writer)
    if args.repeat is None:
        if args.fgd:
            outfile = os.path.join(outpath, f'accuracy_seal_rho_{args.rho}_eta_{args.eta}_fgd_{args.fgd_coef}_GC.csv')
        else:
            outfile = os.path.join(outpath, f'accuracy_seal_rho_{args.rho}_eta_{args.eta}_GC.csv')
    else:
        if args.fgd:
            outfile = os.path.join(outpath, f'{args.repeat}_accuracy_seal_rho_{args.rho}_eta_{args.eta}_fgd_{args.fgd_coef}_GC.csv')
        else:
            outfile = os.path.join(outpath, f'{args.repeat}_accuracy_seal_rho_{args.rho}_eta_{args.eta}_GC.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--alg', type=str, default='SEAL',
                        help='Name of algorithms.')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for graph classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=16)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./OneDSOutputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default=None, choices=['chem', 'biochem', 'biochemsn', 'biosncv'])
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
    parser.add_argument('--n_rw', type=int, default=16,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16,
                        help='Size of position encoding (max degree).')
    parser.add_argument('--n_ones', type=int, default=16,
                        help='Size of position encoding (ones).')
    parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])
    
    parser.add_argument('--modelpath', type=str, default='./OneDSmodel',
                        help='The path of global model saved.')
    
    parser.add_argument('--OneDataset', help='specify the dataset of one domain',
                        type=str, default='COLLAB', choices=['NCI1', 'PROTEINS', 'IMDB-BINARY', 'COLLAB'])
    parser.add_argument('--convert_x', type=bool, default=False,
                        help='whether to convert original node features to one-hot degree features')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='number of clients')
    parser.add_argument('--partition', type=str, default='non-iid',
                         help='whether partitioned data is iid or non-iid')
    parser.add_argument('--alpha', type=float, default=0.01,
                         help='the parameter of dirichlet distribution for non-iid partition')
    parser.add_argument('--standardize', type=bool, default=False,
                        help='whether to standardize the distance matrix')

    parser.add_argument('--mu', type=float, default=0.01,
                        help='the coefficient of proximal term in FedProx.')
    parser.add_argument('--epsilon1', type=float, default=0.01,
                        help='the threshold epsilon1 for GCFL.')
    parser.add_argument('--epsilon2', type=float, default=0.1,
                        help='the threshold epsilon2 for GCFL.')
    
    parser.add_argument('--rho', type=float, default=0.1,
                        help='rho for SEAL.')
    parser.add_argument('--eta', type=float, default=0.01,
                        help='eta for SEAL.')
    
    parser.add_argument('--fgd', action='store_true',
                        help='whether to use Federated Graph Decorr')
    parser.add_argument('--fgd_coef', type=float, default=0.1,
                        help='coefficient of the Federated Graph Decorr loss')
    
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 16
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    outpath = os.path.join(args.outbase, 'raw', args.OneDataset)
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    if args.data_group == None:
        modelpath = os.path.join(args.modelpath, args.OneDataset, args.alg)
    if args.OneDataset == None:
        modelpath = os.path.join(args.modelpath, args.data_group, args.alg)
    Path(modelpath).mkdir(parents=True, exist_ok=True)
    print(f"Model Path: {modelpath}")

    print("Preparing data ...")
    splitedData, df_stats = setupGC.prepareData_oneDS(args, args.datapath, args.OneDataset, args.num_clients, args.batch_size, args.convert_x, seed=seed_dataSplit)
    print("Done")

    if args.repeat is None:
        outf = os.path.join(outpath, 'stats_trainData.csv')
    else:
        outf = os.path.join(outpath, f'{args.repeat}_stats_trainData.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    args.n_se = args.n_rw + args.n_dg

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    if 'fedstar' in args.alg:
        sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.OneDataset}_{args.alg}_{args.type_init}_{args.repeat}')
    else:
        if args.fgd:
            sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.OneDataset}_{args.alg}_fgd_{args.repeat}')
        else:
            sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.OneDataset}_{args.alg}_{args.repeat}')
    summary_writer = SummaryWriter(sw_path)

    if args.alg == 'SEAL':
        process_seal(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), summary_writer=summary_writer)