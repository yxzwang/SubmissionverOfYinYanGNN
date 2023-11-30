import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.dataloading.negative_sampler import Uniform
import datasets
import dgl.data
import random
from small_graph_parser import parse
args = parse()
# random seed

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
torch.cuda.current_device()
torch.cuda._initialized = True

# load dataset
if args.dataset == 'cora':
    dataset = dgl.data.CoraGraphDataset()
elif args.dataset == 'citeseer':
    dataset = dgl.data.CiteseerGraphDataset()
elif args.dataset == 'pubmed':
    dataset = dgl.data.PubmedGraphDataset()

else:
    raise NotImplementedError('dataset not exist!')

g = dataset[0]

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    # edge split  train/valid/test 0.7/0.1/0.2
    # for distillation train/valid/test 0.8/0.05/0.15
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    valid_size = int(len(eids) * 0.1)
    test_size = int(len(eids) * 0.2)
    train_size = g.number_of_edges() - valid_size - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    valid_pos_u, valid_pos_v = u[eids[test_size: test_size + valid_size]], v[eids[test_size: test_size + valid_size]]
    train_pos_u, train_pos_v = u[eids[test_size + valid_size:]], v[eids[test_size + valid_size:]]

    # find negative edges
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    # negative edge split test/valid/train positive edge: negative edge = 1:1
    # neg_eids = np.random.choice(len(neg_u), int(0.7 * g.number_of_edges()), replace=True)
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges(), replace=True)
    test_neg_u, test_neg_v = (neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]])
    valid_neg_u, valid_neg_v = (neg_u[neg_eids[test_size: test_size + valid_size]], neg_v[neg_eids[test_size: test_size + valid_size]])
    train_neg_u, train_neg_v = (neg_u[neg_eids[test_size + valid_size:]], neg_v[neg_eids[test_size + valid_size:]])

    # train procedure: remove edges
    train_g = dgl.remove_edges(g, eids[:test_size + valid_size])
    train_g = dgl.to_bidirected(train_g, copy_ndata=True)
    train_g = dgl.add_self_loop(train_g)

    valid_g = dgl.remove_edges(g, eids[:test_size])
    valid_g = dgl.to_bidirected(valid_g, copy_ndata=True)
    valid_g = dgl.add_self_loop(valid_g)


    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    valid_pos_g = dgl.graph((valid_pos_u, valid_pos_v), num_nodes=g.number_of_nodes())
    valid_neg_g = dgl.graph((valid_neg_u, valid_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

else:

    split_edge = dataset.get_edge_split()
    index = torch.randperm(split_edge['train']['edge'].size(0))
    index = index[: split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][index]}

    train_pos_u, train_pos_v = split_edge['train']['edge'].t()
    # needs global uniform()
    train_neg_u, train_neg_v = split_edge['eval_train']['edge'].t()
    valid_pos_u, valid_pos_v = split_edge['valid']['edge'].t()
    valid_neg_u, valid_neg_v = split_edge['valid']['edge_neg'].t()
    test_pos_u, test_pos_v = split_edge['test']['edge'].t()
    test_neg_u, test_neg_v = split_edge['test']['edge_neg'].t()


# Dot predictor and MLP predictor
import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = edges.src['h'] * edges.dst['h']
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class DistPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W = nn.Linear(h_feats, h_feats)

        H = torch.rand(h_feats, h_feats)
        bound = 1 / h_feats  # normal
        nn.init.normal_(H, 0, bound)
        H = H + torch.eye(h_feats)

        self.H = nn.Parameter(H)

    def apply_edges(self, edges):
        # norm_s = torch.norm(edge.src['h'])
        # norm_t = torch.norm(edge.dst['h'])
        # h = ((F.normalize(edges.src['h'], dim=-1) - F.normalize(edges.dst['h']@ self.H, dim=-1)) ** 2).sum(-1)
        h = ((edges.src['h'] - edges.dst['h'] @ self.H) ** 2).sum(-1)
        # h = ((edges.src['h'] - edges.dst['h']) ** 2).sum(-1)
        return {'score': h}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = self.W(h)
            g.apply_edges(self.apply_edges)
            return g.edata['score']

# metric
metric = args.metric
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(scores.device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_loss_no_sigmoid(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    print('postive score: {}'.format(pos_score[0]))
    print('negative score: {}'.format(neg_score[0]))
    labels = torch.cat([torch.ones(pos_score.shape[0]), -torch.ones(neg_score.shape[0])]).to(scores.device)
    return (scores * labels).sum(-1)

def compute_loss_log_sigmoid(pos_score, neg_score, gamma):
    pos_score = gamma - pos_score
    neg_score = neg_score - gamma
    scores = torch.cat([pos_score, neg_score])
    return -F.logsigmoid(scores).sum(-1)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def compute_hr(pos_score, neg_score, k):
    # hits @ k:
    # cora: hits @ 100 citeseer: hits@100 pubmed hits@100
    # collab hits @ 50 PPA hits @ 100 DDI hits @ 20
    if len(neg_score) < k:
        return 1.0
    kth_scores = torch.topk(neg_score, k)[0][-1]
    hitsk = float(torch.sum(pos_score > kth_scores).cpu()) / len(pos_score)
    return hitsk
    
# model
from model.model import UnfoldingKGE

from dgl.nn.pytorch.conv import GraphConv
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat) 
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

from dgl.nn.pytorch.conv import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
from dgl.nn.pytorch.conv import GATConv
class GAT(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, 1)
        self.conv2 = GATConv(h_feats, h_feats, 1)
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h.squeeze()

from dgl.nn.pytorch.conv import GINConv
class GIN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GIN, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(in_feats, h_feats), 
            nn.ReLU(),
            nn.Linear(h_feats, h_feats))
        self.MLP2 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.conv1 = GINConv(self.MLP1, aggregator_type='sum')
        self.conv2 = GINConv(self.MLP2, aggregator_type='sum') 
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # h = self.conv2(g, h)
        return h

class JKnet(nn.Module):
    pass

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(MLP, self).__init__() 
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        # self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, in_feat):
        # in_feat = self.dropout(in_feat)
        h = self.linear1(in_feat)
        h = F.relu(h)
        h = self.linear2(h)
        return h 


        
drop_out = nn.Dropout(p=0.5)

if args.model == 'yinyang' or args.model == 'yinyang_l':
    model = UnfoldingKGE(args, in_size = train_g.ndata['feat'].shape[1], hid_size = args.hidden, model=args.model)
elif args.model == 'SAGE':
    model = GraphSAGE(in_feats = train_g.ndata['feat'].shape[1], h_feats = args.hidden)
elif args.model == 'GAT':
    model = GAT(in_feats = train_g.ndata['feat'].shape[1], h_feats = args.hidden)
elif args.model == 'GCN':
    model = GCN(in_feats = train_g.ndata['feat'].shape[1], h_feats = args.hidden)
elif args.model == 'JKnet':
    pass   
elif args.model == 'GIN':
    model = GIN(in_feats = train_g.ndata['feat'].shape[1], h_feats = args.hidden)
elif args.model == 'mlp':
    model = MLP(in_feats = train_g.ndata['feat'].shape[1], h_feats = args.hidden)
elif args.model == 'node2vec':
    # run a node2vec embedding 
    path = str(args.dataset) + '.pt'
    node2vec = torch.load(path, map_location='cpu')
    in_feat = torch.cat([train_g.ndata['feat'], node2vec], dim=1)
    model = MLP(in_feats = in_feat.shape[1], h_feats = args.hidden)
elif args.model == 'MF':
    emb = torch.nn.Embedding(g.num_nodes(), args.hidden)
else:
    raise NotImplementedError
    
# pred = MLPPredictor(args.hidden)
# pred = DotPredictor()

if args.score_func == 'Hadamard':
    pred = Hadamard_MLPPredictor(args.hidden)
elif args.score_func == 'dist':
    pred = DistPredictor(args.hidden)

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
# device="cpu"
if args.model != 'MF':
    model = model.to(device)
else: 
    emb = emb.to(device)
pred = pred.to(device)
train_pos_g=train_pos_g.to(device)
train_neg_g=train_neg_g.to(device)
valid_pos_g=valid_pos_g.to(device)
valid_neg_g=valid_neg_g.to(device)
test_pos_g=test_pos_g.to(device)
test_neg_g=test_neg_g.to(device)
train_g=train_g.to(device)
valid_g=valid_g.to(device)
if args.model == 'MF':
    optimizer = torch.optim.Adam(itertools.chain(emb.parameters(), pred.parameters()), lr=args.lr)
else:
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)


# ----------- 4. training -------------------------------- #

best_emb = None
best_model = None
best_epoch = 0
best_dev_auc = 0.0
best_dev_hits100 = 0.0

from sklearn.metrics import roc_auc_score

for epoch in range(1000):
    # forward

    if args.model == 'yinyang':
        h = model.forward_small(graph = train_g, x = train_g.ndata["feat"].to(device))
    elif args.model == 'yinyang_l': 
        h = model.forward_small(graph = train_g, x = train_g.ndata["feat"].to(device))
    elif args.model == 'mlp': 
        h = model(in_feat = train_g.ndata['feat']).to(device)
    elif args.model == 'node2vec':
        in_feat = in_feat.to(device)
        h = model(in_feat = in_feat).to(device) 
    elif args.model == 'MF': 
        h = emb.weight
        drop_out.train()
        h = drop_out(h)
    elif args.model == 'neggnn':
        sampler = negative_sampler(train_g, torch.LongTensor(range(train_g.num_edges())).to(device))
        neg_train_g = dgl.graph(sampler, device="cpu")
        neg_train_g = dgl.to_bidirected(neg_train_g, copy_ndata=True).to(device)
        h = model(pos_g=train_g, neg_g=neg_train_g, in_feat=train_g.ndata["feat"].to(device))
    else:
        h = model(g = train_g, in_feat = train_g.ndata["feat"]).to(device)

    pos_score = pred(train_pos_g, h)
    if args.dynamic == 'static':
        neg_score = pred(train_neg_g, h)
    else:
        train_neg_eids = np.random.choice(len(neg_u), int(0.7 * g.number_of_edges()), replace=True)
        train_neg_u, train_neg_v = neg_u[train_neg_eids], neg_v[train_neg_eids]
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes()).to(device)
        neg_score = pred(train_neg_g, h)

    loss = compute_loss(pos_score, neg_score)
    # loss = compute_loss_no_sigmoid(pos_score, neg_score)
    # loss = compute_loss_log_sigmoid(pos_score, neg_score, 12)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():

        pos_score = pred(valid_pos_g, h)
        neg_score = pred(valid_neg_g, h)
        dev_hits100 = compute_hr(pos_score, neg_score, 100)
        # dev_hits100 = compute_hr(pos_score, neg_score, 20)
        # dev_auc = compute_auc(pos_score, neg_score)

        if metric == 'hits@k' and dev_hits100 > best_dev_hits100:
            best_dev_hits100 = dev_hits100
            if args.model == 'MF':
                best_emb = emb.state_dict()
                best_epoch = epoch 
            else:
                best_model = model.state_dict()
                best_epoch = epoch

        if metric == 'auc' and dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            best_model = model.state_dict()
            best_epoch = epoch

        if epoch - best_epoch >= 100:
            break

        # print('epoch: {}, dev_hits100: {}, loss: {}'.format(epoch, dev_hits100, loss))
        print('epoch: {}, dev_hits100: {}, loss: {}'.format(epoch, dev_hits100, loss))

if metric == 'hits@k':
    print('best epoch: {}, best_dev_hits100: {}'.format(best_epoch, best_dev_hits100))
elif metric == 'auc':
    print('best epoch: {}, best_dev_auc: {}'.format(best_epoch, best_dev_auc))

# ----------- 5. check results ------------------------ #


with torch.no_grad():
    
    if args.model == 'yinyang':
        model.load_state_dict(best_model)
        inference_h = model.forward_small(graph = valid_g, x = train_g.ndata["feat"].to(device))
    elif args.model == 'yinyang_l': 
        model.load_state_dict(best_model)
        inference_h = model.forward_small(graph = valid_g, x = train_g.ndata["feat"].to(device))
    elif args.model == 'mlp':
        model.load_state_dict(best_model)
        inference_h = model(in_feat = train_g.ndata['feat']).to(device)
    elif args.model == 'MF':
        emb.load_state_dict(best_emb)
        inference_h = emb.weight.to(device)
        drop_out.eval()
        inference_h = drop_out(inference_h)
    elif args.model == 'node2vec':
        model.load_state_dict(best_model)
        in_feat = in_feat.to(device)
        inference_h = model(in_feat = in_feat).to(device)
    elif args.model == 'neggnn':
        model.load_state_dict(best_model)
        sampler = negative_sampler(valid_g, torch.LongTensor(range(valid_g.num_edges())).to(device))
        neg_valid_g = dgl.graph(sampler, device="cpu")
        neg_valid_g = dgl.to_bidirected(neg_valid_g, copy_ndata=True).to(device)
        inference_h = model(pos_g = valid_g, neg_g = neg_valid_g, in_feat = valid_g.ndata["feat"].to(device))
    else:
        model.load_state_dict(best_model)
        inference_h = model(g = valid_g, in_feat = train_g.ndata["feat"].to(device))

    pos_score = pred(test_pos_g, inference_h)
    neg_score = pred(test_neg_g, inference_h)

    # print("AUC", compute_auc(pos_score, neg_score))
    print("hits100", compute_hr(pos_score, neg_score, 100))
    # print("hits100", compute_hr(pos_score, neg_score, 20))




