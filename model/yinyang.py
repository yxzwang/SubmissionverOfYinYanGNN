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

def normalized_AX(graph, X):
    """Y = D^{-1/2}AD^{-1/2}X"""
    Y = D_power_X(graph, X, -0.5)  # Y = D^{-1/2}X
    Y = AX(graph, Y)  # Y = AD^{-1/2}X
    Y = D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}AD^{-1/2}X
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
    norm[norm == float('inf')] = 0
    norm[norm == float('-inf')] = 0 
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
    return degs.view(-1, 1)

        

class Propagate(nn.Module):
    def __init__(self):
        super().__init__()

    def prop(self, graph, Y, lam):

        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        Y = AX(graph, Y)
        Y = D_power_bias_X(graph, Y, -0.5, lam, 1 - lam)
        return Y

    def forward(self, graph, Y, X, alp, lam0, lam, lam_K, K, g_list, D_list):
        if g_list:
            D_0, D_1 = D_list[0], D_list[1]
            assert Y.shape[0] == normalized_AX(graph[0], X).shape[0]
            assert Y.shape[0] == normalized_AX(graph[1], X).shape[0]
            Y_new = Y - alp * ( lam0 * (Y - X) / (D_0 + D_1) + lam * (Y - normalized_AX(graph[0], Y))- lam_K * 1/K * (Y - normalized_AX(graph[1], Y)))
            # Y_new = Y - alp * ( lam0 * (Y - X) + lam * (Y - normalized_AX(graph[0], Y))- lam_K * 1/K * (Y - normalized_AX(graph[1], Y)))
            return Y_new
        else:
            D_0 = D_list[0]
            return Y - alp * ( lam0 * (Y - X) / (D_0) + lam * (Y - normalized_AX(graph, Y)))

class UnfoldindAndAttention(nn.Module):
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
        self.lam0 = args.lam0
        self.K = args.K
        self.lam_K = args.lam_K
        
        self.gamma       = args.gamma
        prop_method      = Propagate 
        self.prop_layers = nn.ModuleList([prop_method() for _ in range(self.prop_step)])
        
        self.init_attn   =  None
        self.attn_layer  =  None
        self.etas        = nn.Parameter(tc.ones(self.d)) if self.use_eta else None

    def forward(self , g , X):
        Y0 = X
        Y = Y0
        if isinstance(g,list): # g=[g_pos,g_neg]
            g_list=True
            
            g[0].edata["w"]    = tc.ones(g[0].number_of_edges(), 1, device = g[0].device)
            g[0].ndata["deg"]  = g[0].in_degrees().float()
            # g[1]=g[1].add_self_loop()
            g[1].edata["w"]    = tc.ones(g[1].number_of_edges(), 1, device = g[1].device)
            g[1].ndata["deg"]  =g[1].in_degrees().float()
            # print("0:",(g[0].ndata["deg"]==0).sum())
            # print("1:",(g[1].ndata["deg"]==0).sum())
            if self.init_att:
                g[0] = self.init_attn(g[0], Y, self.gamma, "pos")
                g[1] = self.init_attn(g[1], Y, self.gamma, "neg")
            D_0, D_1 = getgraphdegree(g[0]), getgraphdegree(g[1])
            if self.args.debug:
                print(f"D0={D_0}, D1={D_1}")
        else: # only positive 
            g_list=False
            g.edata["w"]    = tc.ones(g.number_of_edges(), 1, device = g.device)
            g.ndata["deg"]  = g.in_degrees().float()
            if self.init_att:
                g = self.init_attn(g, Y, self.gamma,"pos")
            # print("0:",(g.ndata["deg"]==0).sum())
            D_0, D_1 = getgraphdegree(g), 0 
        for k, layer in enumerate(self.prop_layers):
            
            Y = layer(g, Y, Y0, self.alp, self.lam0, self.lam, self.lam_K, self.K,g_list, [D_0, D_1])
            # add dropout

            # if self.init_att:
            #     g[0] = self.init_attn(g[0], Y, self.gamma,"pos")
            #     g[1] = self.init_attn(g[1], Y, self.gamma,"neg")
        # import ipdb
        # ipdb.set_trace()
        return Y
    def compute_E0(self,Y,Y0):
        return th.norm(Y-Y0)**2
    def compute_E1(self,g,Y):
        u,v=g.edges()
        Yu=Y[u]
        Yv=Y[v]
        energy=th.sum((Yu-Yv)**2)
        return energy
    def compute_E2(self,g,Y):
        u,v=g.edges()
        Yu=Y[u]
        Yv=Y[v]
        
        return ((Yu-Yv)**2).sum(dim=-1)

   