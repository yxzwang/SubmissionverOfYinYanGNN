import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--alp", default=1, type=float, help="step size for inner loop")
    parser.add_argument("--lam", default=1, type=float, help="structure reg")
    parser.add_argument("--K", default=0, type=int, help="negative sample numbers for each edge")
    parser.add_argument("--nomlp", default=False, action='store_true')
    parser.add_argument("--dataset", default='cora', choices=['cora', 'citeseer', 'pubmed', 'ogbl-collab'], type=str)
    parser.add_argument("--model", default='yinyang', type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--hidden", default=16, type=int)
    parser.add_argument("--metric", default='hits@k', choices=['hits@k', 'auc', 'ap'], type=str)
    parser.add_argument("--score_func", default='Hadamard', type=str)
    parser.add_argument("--dynamic", default='static', choices=['static', 'dynamic'], type=str)
    parser.add_argument("--linear", action='store_true', default=False)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lam0", default=0, type=float, help='coefficient for mlp')
    parser.add_argument("--lam_K", default=1, type=float, help='coefficient for negative edge')
    parser.add_argument("--gamma", default=99, type=float, help='gamma for logsigmoid')
    parser.add_argument("--sign_k", default=0, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--gnn_k", default=0, type=int)
    parser.add_argument("--uniform", default=False, action='store_true')
    parser.add_argument("--learnable", default=False, action='store_true')
    parser.add_argument("--split_negs", default=False, action='store_true')
    
    args = parser.parse_args()
    args.mlp = not args.nomlp
    return args


