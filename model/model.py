##dgl 
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler,ShaDowKHopSampler
from .yinyang import UnfoldindAndAttention
from .yinyang_l import UnfoldindAndAttention_l
import tqdm
from dgl.dataloading.negative_sampler import Uniform, GlobalUniform
# from utils.sampler import Uniform_and_bidirect
from .yinyang import normalized_AX

class SIGNEmbedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNEmbedding, self).__init__()

        self.K = K
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.K + 1):
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lin_out = nn.Linear((K + 1) * hidden_channels, out_channels)
        self.dropout = dropout
        self.adj_t = None

    def reset_parameters(self):
        for lin, bn in zip(self.lins, self.bns):
            lin.reset_parameters()
            bn.reset_parameters()

    def forward(self, x, adj_t):
        adj_t.ndata["deg"]  = adj_t.in_degrees().float()
        adj_t.edata["w"]    = torch.ones(adj_t.number_of_edges(), 1, device = adj_t.device)
        hs = []
        for lin, bn in zip(self.lins, self.bns):
            h = lin(x)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
            x = normalized_AX(adj_t , x)
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x
    
class MLP(nn.Module):


    def __init__(self, args, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = args.dropout

    def forward(self, x):
        h = x
        h = self.linears[0](h)
        h = self.batch_norm(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        h = self.linears[1](h)
        return h

class UnfoldingKGE(nn.Module):
    def __init__(self, args, in_size, hid_size,use_soft=True,a=True,b=True,c=True,use_linear=False,reduce="mean",model="distmult", num_nodes=None):
        super().__init__()
        self.split_negs = args.split_negs
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        if args.linear:
            self.mlp=nn.ModuleList([nn.Linear(in_size, hid_size), nn.Dropout(args.dropout)])
        else:
            self.mlp = MLP(args, in_size, hid_size, hid_size)
    
        if model=="yinyang":
            self.layer=UnfoldindAndAttention(args,False,1,1,1,False,False,0.5,True)
        elif model=="yinyang_l":
            self.layer=UnfoldindAndAttention_l(args,False,1,1,1,False,False,0.5,True)
        else:
            print("not implement")
        if args.sign_k > 0:
            self.sign_emb = SIGNEmbedding(hid_size, hid_size, hid_size, args.sign_k, dropout=args.sign_dropout)
            self.node_emb = nn.Embedding(num_nodes ,hid_size)
        self.hid_size = hid_size
        self.args = args 
        if args.K > 0:
            if self.split_negs:
                self.neg_sampler = Uniform(1)
            else:
                self.neg_sampler = Uniform(args.K) #negative samples > 0
        else:
            self.neg_sampler = None
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, graph, x):
        h = self.mlp(x)
        if self.neg_sampler:
            u, v = self.neg_sampler(graph,torch.arange(graph.num_edges(),device=x.device))
            neg_graph = dgl.graph((v,u),device=graph.device)
            h = self.layer.forward([graph,neg_graph], h)
        else:
            h = self.layer.forward(graph, h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def forward_small(self, graph, x):
        h = self.mlp(x)
        if self.neg_sampler is not None:

            if not self.args.split_negs:
                neg_sampler = GlobalUniform(self.args.K)
                sampler = neg_sampler(graph, torch.LongTensor(range(graph.num_edges())).to(x.device))
                neg_graph = dgl.graph(sampler, device="cpu", num_nodes=graph.number_of_nodes())
                neg_graph = dgl.to_bidirected(neg_graph, copy_ndata=True).to(x.device)
                h = self.layer.forward([graph, neg_graph], h)
            else:
                neg_graphs = []
                for i in range(self.args.K):
                    sampler = self.neg_sampler(graph, torch.LongTensor(range(graph.num_edges())).to(x.device))
                    neg_graph = dgl.graph(sampler, device="cpu")
                    neg_graph = dgl.to_bidirected(neg_graph, copy_ndata=True).to(x.device)
                    neg_graphs.append(neg_graph)
                h = self.layer.forward([graph, neg_graphs], h)
        else:
            h = self.layer.forward(graph, h)
        return h
    
    def forward_no_sampler(self, graph, x, neg_graph=None):
        
        if self.args.sign_k > 0:
            sign_emb = self.sign_emb(self.node_emb.weight, graph)
            x =sign_emb
        h = self.mlp(x)
        if neg_graph is not None:
            h = self.layer.forward([graph, neg_graph], h)
        else:
            h = self.layer.forward(graph, h)
        
        return h
    
    