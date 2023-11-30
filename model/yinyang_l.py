import pdb
import random
from functools import partial
import dgl
import torch as tc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init

def normalized_AX(graph, X, D=None):
    """Y = D^{-1/2}AD^{-1/2}X"""
    if D is None:
        Y = D_power_X(graph, X, -0.5)  # Y = D^{-1/2}X
        Y = AX(graph, Y)  # Y = AD^{-1/2}X
        Y = D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}AD^{-1/2}X
    else:
        Y = X * th.pow(D, -0.5)
        Y = AX(graph, Y)  # Y = AD^{-1/2}X
        Y = Y * th.pow(D, -0.5)

    return Y

def AX(graph, X):
    """Y = AX"""
    graph.srcdata["h"] = X
    graph.update_all(
        fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"),
    )
    Y = graph.dstdata["h"]

    return Y


def D_power_X(graph, X, power):
    """Y = D^{power}X"""

    degs = graph.ndata["deg"]
    norm = th.pow(degs, power)
    Y = X * norm.view(X.size(0), 1)
    return Y

def D_power_bias_X(graph, X, power, coeff, bias):
    """Y = (coeff*D + bias*I)^{power} X"""
    degs = graph.ndata["deg"]
    degs = coeff * degs + bias
    norm = th.pow(degs, power)
    Y = X * norm.view(X.size(0), 1)
    return Y
def getgraphdegree(graph):
    degs = graph.ndata["deg"]
    # norm = th.pow(degs, -1)
    
    return degs.view(-1,1)

        

class Propagate(nn.Module):
    def __init__(self):
        super().__init__()

    def prop(self, graph, Y, lam):

        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        Y = AX(graph, Y)
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)

        return Y

    def forward(self, graph, Y, X, alp, lam0,lam, lam_K, K, g_list, D_list):
        if g_list:
            D_0 = D_list[0]
            D_1 = th.sum(th.stack(D_list[1:],dim=-1),dim=-1)
            Y_new = Y - alp * ( lam0 * (Y-X) / (D_0 + D_1) + lam * (Y - normalized_AX(graph[0], Y)))
            for i in range(len(graph[1])):
                Y_new = Y_new + alp * th.sigmoid(lam_K[i]) * 1/K  * (Y - normalized_AX(graph[1][i], Y, D_1))
            return Y_new
        else:
            return Y - alp * ( lam0 * (Y-X) / D_list[0]  + lam * (Y - normalized_AX(graph, Y)))

class UnfoldindAndAttention_l(nn.Module):
    def __init__(self,args, attn_aft, swish, T, p, use_eta, init_att , attn_dropout, precond):

        super().__init__()
        self.args=args
        self.d      = args.hidden
        self.alp    = args.alp if args.alp > 0 else 1 / (args.lam + 1) # automatic set alpha
        self.lam    = args.lam
        self.swish    = swish
        self.p      = p
        self.prop_step = args.prop_step
        self.attn_aft  = attn_aft
        self.use_eta   = use_eta
        self.init_att  = init_att
        self.lam0=args.lam0
        self.K=args.K
        if args.split_negs:
            # self.lam_K = th.ones(args.K)
            self.lam_K = nn.Parameter(th.randn(args.K, requires_grad=True))
        else:
            self.lam_K = args.lam_K
        
        self.gamma       = args.gamma
        prop_method      = Propagate 
        self.prop_layers = nn.ModuleList([prop_method() for _ in range(self.prop_step)])


    def forward(self , g , X):
        Y0=X
        # Y0 = F.normalize(X,dim=-1,p=2)
        Y=Y0
        if isinstance(g,list):##g=[g_pos,g_neg]
            g_list=True
            g[0].edata["w"]    = tc.ones(g[0].number_of_edges(), 1, device = g[0].device)
            g[0].ndata["deg"]  = g[0].in_degrees().float()
            if self.args.split_negs:
                D_list=[getgraphdegree(g[0])]
                for i in range(self.args.K):
                    g[1][i].edata["w"]    = tc.ones(g[1][i].number_of_edges(), 1, device = g[1][i].device)
                    g[1][i].ndata["deg"]  = g[1][i].in_degrees().float()
                    D_list.append(getgraphdegree(g[1][i]))
            else:
                g[1].edata["w"]    = tc.ones(g[1].number_of_edges(), 1, device = g[1].device)
                g[1].ndata["deg"]  =g[1].in_degrees().float()
                # print("0:",(g[0].ndata["deg"]==0).sum())
                # print("1:",(g[1].ndata["deg"]==0).sum())
                D_0=getgraphdegree(g[0])
                D_1=getgraphdegree(g[1])
                D_list=[D_0,D_1]
                # if self.args.debug:
                #     print(f"D0={D_0}, D1={D_1}")
        else:##only positive 
            g_list=False
            try:
                g.edata["w"]    = g.edata["weight"].float()
            except:
                g.edata["w"]    = tc.ones(g.number_of_edges(), 1, device = g.device)
            g.ndata["deg"]  = g.in_degrees().float()
            if self.init_att:
                g = self.init_attn(g, Y, self.gamma,"pos")
            # print("0:",(g.ndata["deg"]==0).sum())
            D_0=getgraphdegree(g)
            D_1=0
            D_list=[D_0, D_1]
        for k, layer in enumerate(self.prop_layers):
            Y = layer(g, Y, Y0, self.alp, self.lam0,self.lam,self.lam_K,self.K,g_list,D_list)
        # print(self.lam_K)
        # print(th.sum(th.sigmoid(self.lam_K)).item())
        return Y

