import torch
import torch.nn as nn
import torch.nn.functional as F
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation
import numpy as np
import math, random


def global_global_loss_(pos_graph, sub_graph, neg_graph, measure='JSD'):
    num_graphs = pos_graph.shape[0]
    # num_nodes = l_enc.shape[0]

    pos_mask = torch.eye(num_graphs).cuda()
    neg_mask = 1 - pos_mask

    res = torch.mm(pos_graph, sub_graph.t())
    neg_res = torch.mm(neg_graph, sub_graph.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg1 = get_negative_expectation(res * neg_mask, measure, average=False)
    E_neg2 = get_negative_expectation(neg_res * pos_mask, measure, average=False)
    E_neg = (E_neg1 + E_neg2).mean()

    return E_neg - E_pos, E_neg1.sum()/num_graphs/(num_graphs-1), E_neg2.sum()/num_graphs, E_pos


def local_global_loss_(g_enc, l_enc, batch, measure='JSD'):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


def adj_loss_(l_enc, g_enc, edge_index, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    adj = torch.zeros((num_nodes, num_nodes)).cuda()
    mask = torch.eye(num_nodes).cuda()
    for node1, node2 in zip(edge_index[0], edge_index[1]):
        adj[node1.item()][node2.item()] = 1.
        adj[node2.item()][node1.item()] = 1.

    res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
    res = (1 - mask) * res
    # print(res.shape, adj.shape)
    # input()

    loss = nn.BCELoss()(res, adj)
    return loss


def shuffling(s):
    index = np.arange(s.size(0))
    random.shuffle(index)
    return s[index]


def cat_mi_loss(score1, score2):
    pos_loss = -torch.nn.functional.softplus(-score1)
    neg_loss = -torch.nn.functional.softplus(-score2)

    return (-pos_loss + neg_loss).mean()
