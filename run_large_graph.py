import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.dataloading.negative_sampler import GlobalUniform,_BaseNegativeSampler
from utils.sampler import Uniform_and_bidirect
from torch.utils.data import DataLoader
from losses import *
import dgl.data
import random
import tqdm
def compute_mrr(predictor, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = predictor.pred(h_src*h_dst).squeeze(-1)
        input_dict = {'y_pred_pos': pred[:,0], 'y_pred_neg': pred[:,1:]}
        rr[start:end] = evaluator.eval(input_dict)['mrr_list']
    return rr.mean()
from large_graph_parser import parse
args = parse()
class Sourceneg(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.

    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates
    :attr:`k` pairs of negative edges ``(u, v')``, where ``v'`` is chosen
    uniformly from all the nodes of type ``dsttype``.  The resulting edges will
    also have type ``(srctype, etype, dsttype)``.

    Parameters
    ----------
    k : int
        The number of negative samples per edge.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.dataloading.negative_sampler.PerSourceUniform(2)
    >>> neg_sampler(g, torch.tensor([0, 1]))
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """

    def __init__(self, k):
        self.k = k

    def __call__(self, g, edges):
        src,dst=edges
        neg_src = torch.reshape(src, (-1, 1)).repeat(1, self.k)
        neg_src = torch.reshape(neg_src, (-1,))
        neg_dst = torch.randint(
        0, g.num_nodes(), (self.k * edges.size(1),), dtype=torch.long).to(src.device)

        return torch.stack(
        (neg_src, neg_dst), dim=-1)
        
def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
        
def adjustlr_exp(optimizer, decay_ratio, epoch, lr):
    lr_ = lr * decay_ratio ** epoch 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
seed = args.seed
batch_size = args.batch_size
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
torch.cuda.current_device()
torch.cuda._initialized = True

print(args)

# load dataset
device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
dataset = DglLinkPropPredDataset(name=args.dataset, root='./dataset/')
g = dataset[0]
g = dgl.to_bidirected(g, copy_ndata=True)
g = g.add_self_loop()
split_edge = dataset.get_edge_split()

# data filter
if dataset.name == 'ogbl-collab':
    # here we use val as input 
    # we strictly follow the rules: 
    # we use validation labels for model training after all the model hyper-parameters are fixed using validation labels 
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= args.filter_year).nonzero(as_tuple=False), (-1, ))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    if args.contain_valid_edge:
        u, v = split_edge['valid']['edge'].t()
        g.add_edges(u, v)
        g.add_edges(v, u)
    if args.train_valid_edge:
        split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], split_edge['valid']['edge']), dim=0)
    in_channels = g.ndata['feat'].size(-1)    
    
if dataset.name == 'ogbl-ppa':
    g.ndata['feat'] = g.ndata['feat'].to(torch.float)

if dataset.name == 'ogbl-ddi':
    in_channels = args.hidden 
    
        
    embedding = torch.nn.Embedding(g.num_nodes(), args.hidden)
    torch.nn.init.xavier_uniform_(embedding.weight)
    g.ndata['feat'] = embedding.weight
    embedding = embedding.to(device)
else:
    if args.sign_k>0:
        in_channels = args.hidden
    else:
        in_channels = g.ndata['feat'].size(-1)
    


# predictor
class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.W1(x)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.W2(x)
        return x.squeeze()
    
    def pred(self, x):
        x = self.W1(x)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.W2(x)
        return x
class Dotpredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        return torch.sum(x,dim=-1).squeeze()
    
    def pred(self, x):
        return torch.sum(x,dim=-1)
if args.score_func=="Hadamard":
    pred = Hadamard_MLPPredictor(args.hidden).to(device)
elif args.score_func=="dot":
    print("using dot predictor")
    pred = Dotpredictor(args.hidden).to(device)

if args.use_distance_feature and args.dataset == 'ogbl-ddi':
    pred = Hadamard_MLPPredictor(args.hidden + 512).to(device)

# model
from dgl.nn.pytorch.conv import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.dropout(h, p=args.dropout, training=self.training)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


from model.model import UnfoldingKGE

if args.model == 'yinyang' or args.model == 'yinyang_l':
    hidden_size = args.hidden
    if args.dataset == 'ogbl-ddi' and args.use_distance_feature and args.as_encoder_input:
        hidden_size = args.hidden + 512
    model = UnfoldingKGE(args, in_size = in_channels, hid_size = args.hidden, model=args.model, num_nodes=g.num_nodes()).to(device)
else:
    raise NotImplementedError

if dataset.name == 'ogbl-ddi':
    parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
else:
    parameter = itertools.chain(model.parameters(), pred.parameters())

optimizer = torch.optim.Adam(parameter, lr=args.lr)

best_model = None
best_epoch = args.update_epoch if args.dataset == 'ogbl-ddi' else 0
best_dev_hits20, best_dev_hits50, best_dev_hits100 = 0.0, 0.0, 0.0
best_test_hits20, best_test_hits50, best_test_hits100 = 0.0, 0.0, 0.0
best_mrr = 0.0 
best_auc = 0.0

# loss

def compute_loss(pos_score, neg_score):
    K = neg_score.size(0) / pos_score.size(0)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(scores.device)
    loss_scores = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    pos_weight = torch.ones(pos_score.shape).to(scores.device)
    neg_weight = (1 / K) * torch.ones(neg_score.shape).to(scores.device)
    weight = torch.cat([pos_weight, neg_weight], dim=0).to(scores.device)
    loss = (weight * loss_scores).mean()
    return loss 

def compute_loss_K(pos_score, neg_score):
    K = neg_score.size(0) / pos_score.size(0)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(scores.device)
    loss_scores = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    loss = loss_scores.mean()
    return loss 
# metric
def compute_hr(pos_score, neg_score, thresholds):
    # hits @ k:
    # collab hits @ 50 
    # PPA hits @ 100 
    # DDI hits @ 20

    results = []
    for k in thresholds:
        if len(neg_score) < k:
            return 1.0
        kth_scores = torch.topk(neg_score, k)[0][-1]
        hitsk = float(torch.sum(pos_score > kth_scores).cpu()) / len(pos_score)
        results.append(hitsk)
    return results

def compute_mrr(predictor, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500):
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = predictor.pred(h_src*h_dst).squeeze(-1)
        input_dict = {'y_pred_pos': pred[:,0], 'y_pred_neg': pred[:,1:]}
        rr[start:end] = evaluator.eval(input_dict)['mrr_list']
        del all_dst
        del h_src
        del h_dst
        del pred
    return rr.mean()

from sklearn.metrics import roc_auc_score
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

if args.uniform:
    negative_graph_sampler = GlobalUniform(args.K) if args.K else None
else:
    if args.split_negs:
        negative_graph_sampler = Uniform_and_bidirect(1) if args.K else None
    else:
        negative_graph_sampler = Uniform_and_bidirect(args.K) if args.K else None

if args.dataset =="ogbl-citation2":
    for name in ['train','valid','test']:
        u=split_edge[name]["source_node"]
        v=split_edge[name]["target_node"]
        split_edge[name]['edge']=torch.stack((u,v),dim=0).t()



best_test_results = 0
for epoch in range(args.epochs):

    if negative_graph_sampler:
        g = g.to('cpu')
        if args.uniform:
            neg_g = negative_graph_sampler(g, torch.LongTensor(range(g.num_edges())))
            neg_g = dgl.graph(neg_g, num_nodes=g.number_of_nodes())
            neg_g = dgl.to_bidirected(neg_g, True)
            neg_g = neg_g.to(device)
        else:
            if args.split_negs:
                neg_g = []
                for i in range(args.K):
                    neg_g.append(negative_graph_sampler(g, torch.LongTensor(range(g.num_edges()))).to(device))
            else:
                neg_g = negative_graph_sampler(g, torch.LongTensor(range(g.num_edges())))
                neg_g = neg_g.to(device)
        g = g.to(device)

    else:
        g = g.to(device)
        neg_g = None

    pos_train_edge = split_edge['train']['edge'].to(device)
    if args.loss_negtype=="source":
        neg_sampler = Sourceneg(args.num_neg)
    else:
        neg_sampler = GlobalUniform(args.num_neg)

    dataloader = DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)

    pred.train()
    model.train()
    results = []
    valid_list=[]
    test_list=[]
    for step, edge_index in enumerate(tqdm.tqdm(dataloader)):
        x = g.ndata["feat"]
        if args.dataset=="ogbl-citation2":
            if step >= 1000:
                break
        if args.model == 'yinyang' or args.model == 'yinyang_l':
            h = model.forward_no_sampler(graph=g, x=x, neg_graph=neg_g).to(device)
        else:
            h = model(g=g, in_feat=x).to(device)

        batch_pos_train_edge = pos_train_edge[edge_index]
        if args.loss_negtype=="source":
            neg_train_edge  = neg_sampler(g, batch_pos_train_edge.t())
            batch_neg_train_edge = neg_train_edge

        else:
            neg_train_edge = neg_sampler(g, batch_pos_train_edge.t()[0])
            neg_train_edge = torch.stack(neg_train_edge, dim=0)
            neg_train_edge = neg_train_edge.t()
            batch_neg_train_edge = neg_train_edge

        if args.dataset == 'ogbl-ddi' and args.use_distance_feature and args.as_decoder_input:
            train_edge_src = torch.cat((h[batch_pos_train_edge[:,0]], distance_feature[batch_pos_train_edge[:,0]]), dim=1)
            train_edge_dst = torch.cat((h[batch_pos_train_edge[:,1]], distance_feature[batch_pos_train_edge[:,1]]), dim=1)
            neg_train_edge_src = torch.cat((h[batch_neg_train_edge[:,0]], distance_feature[batch_neg_train_edge[:,0]]), dim=1)
            neg_train_edge_dst = torch.cat((h[batch_neg_train_edge[:,1]], distance_feature[batch_neg_train_edge[:,1]]), dim=1)
            pos_score = pred(train_edge_src, train_edge_dst)
            neg_score = pred(neg_train_edge_src, neg_train_edge_dst)
            
        else:
            pos_score = pred(h[batch_pos_train_edge[:,0]], h[batch_pos_train_edge[:,1]])
            neg_score = pred(h[batch_neg_train_edge[:,0]], h[batch_neg_train_edge[:,1]])

        if args.train_results:
            if args.dataset =="ogbl-ppa":
                train_hits20, train_hits50, train_hits100 = compute_hr(pos_score, neg_score, [20, 50, 100])
                results.append(train_hits100)
                
        margin = None
        num_neg = args.num_neg

        loss = compute_loss(pos_score, neg_score)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.dataset == "ogbl-ddi":
            torch.nn.utils.clip_grad_norm_(x, 1.0)
        elif args.dataset=="ogbl-citation2" or args.dataset=="ogbl-collab":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)

    if args.step_lr_decay and epoch % args.interval == 0:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
        
    if args.exp_lr_decay:
        adjustlr_exp(optimizer, 0.99, epoch, args.lr)
    # print(loss)
    if args.train_results:
        if args.dataset =="ogbl-ppa":
            print(f"average hits100 :{np.mean(results)}")
    
    # gpu = device
    pred.eval()
    model.eval()
    with torch.no_grad():

        if args.model == 'yinyang' or args.model == 'yinyang_l':
            h = model.forward_no_sampler(graph=g, x=g.ndata['feat'], neg_graph=neg_g).to(device)
        else:
            h = model(g=g, in_feat=g.ndata['feat']).to(device)
            
        if args.dataset=="ogbl-citation2":
            results = []
            evaluator = Evaluator(name='ogbl-citation2')
            for split in ['valid', 'test']:
                src = split_edge[split]['source_node'].to(device)
                dst = split_edge[split]['target_node'].to(device)
                neg_dst = split_edge[split]['target_node_neg'].to(device)
                results.append(compute_mrr(pred, evaluator, h, src, dst, neg_dst, device))
            valid_mrr, test_mrr = results
            print('epoch: {}, Valid_MRR: {}, Test_MRR: {}, loss: {}'.format(epoch, valid_mrr, test_mrr, loss))
            del src
            del dst
            del neg_dst
            del h
            torch.cuda.empty_cache()
            
            valid_list.append(valid_mrr)
            test_list.append(test_mrr)
            if args.wandb:
                wandb.log({"Train_Valid_MRR" : valid_mrr, 
                           "Train_Test_MRR" : test_mrr})
                
            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                best_model = model.state_dict()
                best_pred = pred.state_dict()
                best_epoch = epoch

            if test_mrr > best_test_results:
                best_test_results = test_mrr

            if epoch - best_epoch > 100:
                break
            
            
        else:
            
            pos_valid_edge = split_edge['valid']['edge'].t()
            neg_valid_edge = split_edge['valid']['edge_neg'].t()
            pos_valid_edge = pos_valid_edge.to(device)
            neg_valid_edge = neg_valid_edge.to(device)

            pos_test_edge = split_edge['test']['edge'].t()
            neg_test_edge = split_edge['test']['edge_neg'].t()
            pos_test_edge = pos_test_edge.to(device)
            neg_test_edge = neg_test_edge.to(device)
            if args.dataset == 'ogbl-ddi' and args.use_distance_feature and args.as_decoder_input:
                valid_edge_src = torch.cat((h[pos_valid_edge[0]], distance_feature[pos_valid_edge[0]]), dim=1)
                valid_edge_dst = torch.cat((h[pos_valid_edge[1]], distance_feature[pos_valid_edge[1]]), dim=1)
                neg_valid_edge_src = torch.cat((h[neg_valid_edge[0]], distance_feature[neg_valid_edge[0]]), dim=1)
                neg_valid_edge_dst = torch.cat((h[neg_valid_edge[1]], distance_feature[neg_valid_edge[1]]), dim=1)
                pos_score = pred(valid_edge_src, valid_edge_dst)
                neg_score = pred(neg_valid_edge_src, neg_valid_edge_dst)
            else: 
                pos_score = pred(h[pos_valid_edge[0]], h[pos_valid_edge[1]])
                neg_score = pred(h[neg_valid_edge[0]], h[neg_valid_edge[1]])
            dev_hits20, dev_hits50, dev_hits100 = compute_hr(pos_score, neg_score, [20, 50, 100])

            if args.dataset == 'ogbl-ddi' and args.use_distance_feature and args.as_decoder_input:
                test_edge_src = torch.cat((h[pos_test_edge[0]], distance_feature[pos_test_edge[0]]), dim=1)
                test_edge_dst = torch.cat((h[pos_test_edge[1]], distance_feature[pos_test_edge[1]]), dim=1)
                neg_test_edge_src = torch.cat((h[neg_test_edge[0]], distance_feature[neg_test_edge[0]]), dim=1)
                neg_test_edge_dst = torch.cat((h[neg_test_edge[1]], distance_feature[neg_test_edge[1]]), dim=1)
                pos_score = pred(test_edge_src, test_edge_dst)
                neg_score = pred(neg_test_edge_src, neg_test_edge_dst)
            else:
                pos_score = pred(h[pos_test_edge[0]], h[pos_test_edge[1]])
                neg_score = pred(h[neg_test_edge[0]], h[neg_test_edge[1]])
            test_hits20, test_hits50, test_hits100 = compute_hr(pos_score, neg_score, [20, 50, 100])


            if dataset.name == 'ogbl-ddi':
                valid_list.append(dev_hits20)
                test_list.append(test_hits20)
                if dev_hits20 > best_dev_hits20 and epoch >= args.update_epoch:
                    best_dev_hits20 = dev_hits20
                    best_test_hits20 = test_hits20
                    best_model = model.state_dict()
                    best_pred = pred.state_dict()
                    best_embedding = embedding.state_dict()
                    best_epoch = epoch

                if test_hits20 > best_test_results:
                    best_test_results = test_hits20

                if epoch - best_epoch > 50:
                    break
                
            elif dataset.name == 'ogbl-ppa':
                valid_list.append(dev_hits100)
                test_list.append(test_hits100)
                if dev_hits100 > best_dev_hits100:
                    best_dev_hits100 = dev_hits100
                    best_model = model.state_dict()
                    best_pred = pred.state_dict()
                    best_epoch = epoch

                if test_hits100 > best_test_results:
                    best_test_results = test_hits100

                if epoch - best_epoch > 100:
                    break
            else:
                valid_list.append(dev_hits50)
                test_list.append(test_hits50)
                if dev_hits50 > best_dev_hits50:
                    best_dev_hits50 = dev_hits50
                    best_test_hits50 = test_hits50
                    best_model = model.state_dict()
                    best_pred = pred.state_dict()
                    best_epoch = epoch

                if test_hits50 > best_test_results:
                    best_test_results = test_hits50

                # if epoch - best_epoch > 100:
                #     # break

            if dataset.name == 'ogbl-ddi':
                print('epoch: {}, dev_hits20: {}, test_hits20: {}, loss: {}'.format(epoch, dev_hits20, test_hits20, loss))
            elif dataset.name == 'ogbl-ppa':
                print('epoch: {}, dev_hits100: {}, test_hits100: {}, loss: {}'.format(epoch, dev_hits100, test_hits100, loss))
            else:
                print('epoch: {}, dev_hits50: {}, test_hits50: {}, loss: {}'.format(epoch, dev_hits50, test_hits50, loss))
                
                
        
        if neg_g is not None:
            del neg_g

# model.eval()
# pred.eval()
import numpy as np
valid_list=np.array(valid_list)
test_list=np.array(test_list)
index=np.argmax(valid_list)
valid_res=valid_list[index]
test_res=test_list[index]
if args.dataset=="ogbl-citation2":
    print('Final Valid_MRR: {}, Test_MRR: {}'.format(valid_res, test_res))
    
elif args.dataset == 'ogbl-collab' and args.wandb:
    wandb.log({
        "best_epoch": best_epoch,
        "best_dev_hits50": best_dev_hits50, 
        "best_test_hits50": best_test_hits50,
        "best_test_results": best_test_results
    })

elif args.dataset == 'ogbl-ddi' and args.wandb:
    wandb.log({
        "best_epoch": best_epoch,
        "best_dev_hits20": best_dev_hits20,
        "best_test_hits20": best_test_hits20,
        "best_test_results": best_test_results
    })            
        
else:

    print('Final Valid_Hits: {}, Test_Hits: {}'.format(valid_res, test_res))
    