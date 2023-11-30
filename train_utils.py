##dgl 
import torch
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, negative_sampler,ShaDowKHopSampler
import tqdm

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

import numpy as np
import wandb
def compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predictor(h_src*h_dst).squeeze(-1)
        input_dict = {'y_pred_pos': pred[:,0], 'y_pred_neg': pred[:,1:]}
        rr[start:end] = evaluator.eval(input_dict)['mrr_list']
    return rr.mean()
def compute_acc(model, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predictor(h_src*h_dst).squeeze(-1)
        input_dict = {'y_pred_pos': pred[:,0], 'y_pred_neg': pred[:,1:]}
        labels=torch.zeros(pred.shape).to(pred.device)
        labels[:,0]=1#set pos to 1
        result=(pred>0)==labels
        result=result.float()
        result=torch.mean(result,dim=-1)
        rr[start:end] = result
    return rr.mean()
def evaluate(device, graph, edge_split, model, batch_size):
    model.eval()
    evaluator = Evaluator(name='ogbl-citation2')
    with torch.no_grad():
        node_emb = model.inference(graph, device, batch_size)
        results = []
        for split in ['valid', 'test']:
            src = edge_split[split]['source_node'].to(node_emb.device)
            dst = edge_split[split]['target_node'].to(node_emb.device)
            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
            results.append(compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device))
    return results
def evaluate_both(device, graph, edge_split, model, batch_size,args):
    model.eval()
    evaluator = Evaluator(name='ogbl-citation2')
    with torch.no_grad():
        node_emb = model.inference(graph, device, batch_size,args)
        results = []
        for split in ['valid', 'test']:
            src = edge_split[split]['source_node'].to(node_emb.device)
            dst = edge_split[split]['target_node'].to(node_emb.device)
            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
            results.append(compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device))
            results.append(compute_acc(model, evaluator, node_emb, src, dst, neg_dst, device))
    return results
def evaluate_acc(device, graph, edge_split, model, batch_size):
    model.eval()
    evaluator = Evaluator(name='ogbl-citation2')
    with torch.no_grad():
        node_emb = model.inference(graph, device, batch_size)
        results = []
        for split in ['valid', 'test']:
            src = edge_split[split]['source_node'].to(node_emb.device)
            dst = edge_split[split]['target_node'].to(node_emb.device)
            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
            results.append(compute_acc(model, evaluator, node_emb, src, dst, neg_dst, device))
    return results
def evaluate_kd(device, graph, edge_split, model, batch_size):
    model.eval()

    with torch.no_grad():
        node_emb = model.inference(graph, device, batch_size)
        results = []
        for split in ['valid', 'test']:
            src = edge_split[split]['source_node'].to(node_emb.device)
            dst = edge_split[split]['target_node'].to(node_emb.device)
            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
            results.append(compute_acc(model,None, node_emb, src, dst, neg_dst, device))
    return results
def train(args, device, g, reverse_eids, train_id, model,edge_split,eval_batch_size,evaluate_hz,metric="acc"):
    # create sampler & dataloader
    lr=args.lr
    train_batch_size=args.batch_size
    step_epoch=args.step_epoch
    if args.evaluate_first:
        valid_mrr, valid_acc,test_mrr,test_acc = evaluate_both(device, g, edge_split, model, eval_batch_size,args)
        print('Validation MRR {:.4f}, Test MRR {:.4f} ; Validation ACC {:.4f}, Test ACC {:.4f}'.format(valid_mrr.item(),test_mrr.item(),valid_acc.item(),test_acc.item()))
        
    sampler = ShaDowKHopSampler([args.neighbor_num]*args.neighbor_layer,prefetch_node_feats=['feat'])
    
    sampler = as_edge_prediction_sampler(
        sampler, exclude='reverse_id', reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1))
    use_uva = (args.mode == 'mixed')
    dataloader = DataLoader(g, train_id, sampler,
        device=device, batch_size=train_batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch=150
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        acclist=[]
        for it, (input_nodes, pair_graph, neg_pair_graph, subgraph) in enumerate(tqdm.tqdm(dataloader,desc="training in epoch")):

            if args.bidirect:
                subgraph=subgraph.to("cpu")
                u,v=subgraph.edges()
                subgraph.add_edges(v,u)
                subgraph=dgl.to_simple(subgraph.add_self_loop())
                subgraph=subgraph.to(device)
            if args.selfloop:
                subgraph=subgraph.add_self_loop()
            x = subgraph.srcdata['feat']
            pos_score, neg_score = model(pair_graph, neg_pair_graph, subgraph, x)

            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            acclist+=((score>0)==labels).float().cpu().numpy().tolist()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

            if (it+1) == step_epoch: break
        print("Epoch {:05d} | Loss {:.4f}|Acc {:.4f}".format(epoch, total_loss / (it+1),np.mean(acclist)))
        wandb.log({"Train_loss":total_loss / (it+1),"Train_acc":np.mean(acclist)})
        if (epoch+1) % evaluate_hz ==0:

            print('Validation/Testing...')
            if metric=="mrr":
                valid_mrr, test_mrr = evaluate(device, g.to(device), edge_split, model, eval_batch_size)
                print('Validation MRR {:.4f}, Test MRR {:.4f}'.format(valid_mrr.item(),test_mrr.item()))
            elif metric=="acc":
                valid_mrr, test_mrr = evaluate_acc(device, g.to(device), edge_split, model, eval_batch_size)
                print('Validation ACC {:.4f}, Test ACC {:.4f}'.format(valid_mrr.item(),test_mrr.item()))
            elif metric=="both":
                valid_mrr, valid_acc,test_mrr,test_acc = evaluate_both(device, g, edge_split, model, eval_batch_size,args)
                print('Validation MRR {:.4f}, Test MRR {:.4f} ; Validation ACC {:.4f}, Test ACC {:.4f}'.format(valid_mrr.item(),test_mrr.item(),valid_acc.item(),test_acc.item()))
                wandb.log({'Valid_mrr':valid_mrr.item(),'Test_mrr':test_mrr.item(),'Valid_acc':valid_acc.item(),'Test_acc':test_acc.item()})
            torch.cuda.empty_cache()
def train_kg(args, device, g,  train_id, model,val_id,test_id,eval_batch_size,evaluate_hz,metric="acc",lr=0.0005,train_batch_size=64,step_epoch=1000):
    # create sampler & dataloader
    sampler = NeighborSampler([5, 5, 5], prefetch_node_feats=['feat'])
    
    sampler = as_edge_prediction_sampler(
        sampler, exclude='self',
        negative_sampler=negative_sampler.Uniform(1))
    use_uva = (args.mode == 'mixed')
    dataloader = DataLoader(
        g, train_id, sampler,
        device=device, batch_size=train_batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    val_dataloader = DataLoader(
        g, val_id, sampler,
        device=device, batch_size=eval_batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    test_dataloader = DataLoader(
        g, test_id, sampler,
        device=device, batch_size=eval_batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epoch=150
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(tqdm.tqdm(dataloader,desc="training in epoch")):
            x = blocks[0].srcdata['feat']
            pos_score, neg_score = model.forward_hetero(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if (it+1) == step_epoch: break
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it+1)))
        if (epoch+1) % evaluate_hz ==0:
            with torch.no_grad():
                print('Validation/Testing...')
                acclist=[]
                for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(tqdm.tqdm(val_dataloader,desc="validation in epoch")):
                    x = blocks[0].srcdata['feat']
                    pos_score, neg_score = model.forward_hetero(pair_graph, neg_pair_graph, blocks, x)
                    score = torch.cat([pos_score, neg_score])
                    pos_label = torch.ones_like(pos_score)
                    neg_label = torch.zeros_like(neg_score)
                    labels = torch.cat([pos_label, neg_label])
                    acclist+=((score>0)==labels).float().cpu().numpy().tolist()
                print(f"Val_acc: {np.mean(acclist)}")
                acclist=[]
                for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(tqdm.tqdm(test_dataloader,desc="test in epoch")):
                    x = blocks[0].srcdata['feat']
                    pos_score, neg_score = model.forward_hetero(pair_graph, neg_pair_graph, blocks, x)
                    score = torch.cat([pos_score, neg_score])
                    pos_label = torch.ones_like(pos_score)
                    neg_label = torch.zeros_like(neg_score)
                    labels = torch.cat([pos_label, neg_label])
                    acclist+=((score>0)==labels).float().cpu().numpy().tolist()
                print(f"test_acc: {np.mean(acclist)}")