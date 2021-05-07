import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, ReLU, GRU, Conv2d

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from infomax import *


class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, times, mode):
        super(Encoder, self).__init__()

        self.mode = mode
        self.lin0 = torch.nn.Linear(num_features, dim)
        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))

        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.feature_conv = Conv2d(2, 1, (1, 1))
        self.layer_conv = Conv2d(3, 1, (1, 1))
        self.num_subgraph = 2**times
        self.subgraph_conv = Conv2d(self.num_subgraph, 1, (1, 1))
        self.dim = dim
        self.set2set = Set2Set(dim, processing_steps=3)
        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)
        self.f1 = FF(2 * dim, 2 * dim)

        self.partition_list = torch.nn.ModuleList()
        self.times = times
        for i in range(self.num_subgraph - 1):
            self.partition_list.append(torch.nn.Sequential(torch.nn.Linear(dim, 2), torch.nn.Softmax(dim=-1)))

        self.mha_trans = torch.nn.Parameter(torch.Tensor(self.num_subgraph, dim, 2), requires_grad=True)

    def forward(self, data, percent, eval=True):
        if eval:
            h_n0 = F.relu(self.lin0(data.x))
            out = h_n0
            h = out.unsqueeze(0)

            feat_map = []
            for i in range(3):
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                m = self.feature_conv(torch.stack([out, m], dim=1).unsqueeze(-1)).squeeze()
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
                feat_map.append(out)

            global_node = self.layer_conv(torch.stack(feat_map, dim=1).unsqueeze(-1)).squeeze()

            global_graph = self.set2set(global_node, data.batch)

            return global_graph, None, None
        else:
            h_n0 = F.relu(self.lin0(data.x))
            neg_hn0 = self.create_neg_n0(h_n0, data.batch)
            h_cat = torch.cat([h_n0, neg_hn0], dim=0)
            out = h_cat
            h = out.unsqueeze(0)
            prob_loss = 0

            edge_index = torch.cat([data.edge_index, data.edge_index + data.edge_index[0, -1] + 1], dim=-1)
            edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
            feat_map = []
            for i in range(3):
                m = F.relu(self.conv(out, edge_index, edge_attr))
                m = self.feature_conv(torch.stack([out, m], dim=1).unsqueeze(-1)).squeeze()
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
                feat_map.append(out)

            global_node = self.layer_conv(torch.stack(feat_map, dim=1).unsqueeze(-1)).squeeze().view(2, -1, self.dim)
            pos_global_node, neg_global_node = global_node[0], global_node[1]
            neg_global_graph = self.set2set(neg_global_node, data.batch)

            if percent:
                mask = self.sampling_subgraph(percent, data.batch)
                sampling_subgraph_list = []
                for i in range(self.num_subgraph):
                    sampling_subgraph_list.append(
                        self.set2set(pos_global_node[mask[i, :]], batch=data.batch[mask[i, :]]))
                sampling_subgraph = torch.stack(sampling_subgraph_list, dim=1).unsqueeze(-1)
                sampling_subgraph = self.subgraph_conv(sampling_subgraph).squeeze()
            elif self.mode=='MH':
                sampling_subgraph, assignment_prob = self.multi_head_subgraph_generation(pos_global_node, data.batch)
                prob_loss = -torch.sum(torch.abs(assignment_prob[:, :, 0] - assignment_prob[:, :, 1]))

            elif self.mode=='TS':
                sampling_subgraph_list, assignment_prob = self.generate_subgraphs(pos_global_node, data.batch,
                                                                                  self.times)
                prob_loss = -torch.sum(torch.abs(assignment_prob[:, :, 0] - assignment_prob[:, :, 1]))
                sampling_subgraph = torch.stack(sampling_subgraph_list, dim=1).unsqueeze(-1)
                sampling_subgraph = self.subgraph_conv(sampling_subgraph).squeeze()

            else:
                assert False, 'wrong parameter for subgraphs'
            pos_global_graph = self.set2set(pos_global_node, data.batch)

            return pos_global_graph, sampling_subgraph, neg_global_graph, prob_loss

    def sampling_subgraph(self, percent, batch):
        mask = torch.stack([torch.zeros_like(batch) for i in range(self.num_subgraph)],
                           dim=0)  # num_sub_graph x num_total_nodes
        for _ in range(self.num_subgraph):
            last_node_num = 0
            for i in range(batch[-1] + 1):
                node_num = batch[batch == i].size(0)
                sample_node_num = int(np.ceil(node_num * percent))
                idx = np.random.choice(node_num, sample_node_num)
                mask[_, idx + last_node_num] = 1
                last_node_num += node_num

        return mask == 1

    def generate_subgraph(self, node_embeddings, partition_nn):
        node_emb_list = []
        assignment_prob = partition_nn(node_embeddings)
        mask = (assignment_prob >= 0.5)[:, 0:1].long()
        mask = mask.expand_as(node_embeddings)
        mask_ = 1 - mask
        node_emb_list = node_emb_list + [node_embeddings * mask, node_embeddings * mask_]
        return node_emb_list, assignment_prob

    def generate_subgraphs(self, node_embeddings, batch, times):
        subgraph_list = []
        assign_prob_list = []
        node_embed_list = []
        for i in range(times):
            if i == 0:
                node_emb_l, assign_prob = self.generate_subgraph(node_embeddings, self.partition_list[0])
                node_embed_list = node_emb_l
                assign_prob_list.append(assign_prob)
            else:
                temp_list = []
                for j in range(len(node_embed_list)):
                    subgraph_l, assign_prob = self.generate_subgraph(node_embed_list[j],
                                                                     self.partition_list[2 ** i + j - 1])
                    temp_list = temp_list + subgraph_l
                    assign_prob_list.append(assign_prob)
                node_embed_list = temp_list
        for i in range(len(node_embed_list)):
            subgraph_list.append(self.set2set(node_embed_list[i], batch))
        assign_prob = torch.stack(assign_prob_list, dim=0)
        return subgraph_list, assign_prob

    def multi_head_subgraph_generation(self, node_embeddings, batch):
        P = torch.softmax(torch.matmul(node_embeddings, self.mha_trans), dim=-1)
        mask = (P >= 0.5)[:, :, 0].long()
        mask = torch.stack([mask for i in range(self.dim)], dim=-1)
        subgraph_node_embed = (node_embeddings * mask).permute(1, 0, 2).unsqueeze(-1)
        subgraph_embed = self.subgraph_conv(subgraph_node_embed).squeeze()
        return self.set2set(subgraph_embed, batch), P

    def create_neg_n0(self, h_n0, batch):
        neg_hn0 = []
        for i in range(int(batch[-1] + 1)):
            mask = batch == i
            h = h_n0[mask, :]
            idx = np.random.choice(h.size(0), h.size(0))
            neg_hn0.append(h[idx])
        return torch.cat(neg_hn0, dim=0)


class Net(torch.nn.Module):
    def __init__(self, num_features, dim, times, mode, use_unsup_loss=False, separate_encoder=False, use_prob_loss=False):
        super(Net, self).__init__()

        self.embedding_dim = dim
        self.num_subgraph = 2 ** times
        self.separate_encoder = separate_encoder

        self.local = True
        # self.prior = False

        self.encoder = Encoder(num_features, dim, times,mode)
        if separate_encoder:
            self.unsup_encoder = Encoder(num_features, dim, times,mode)
            self.ff1 = FF(2 * dim, dim)
            self.ff2 = FF(2 * dim, dim)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc2 = torch.nn.Linear(dim, 1)

        if use_unsup_loss:
            self.local_d = FF(2 * dim, dim)
            self.global_d = FF(2 * dim, dim)

        self.use_prob_loss = use_prob_loss

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, percent=None):
        global_graph, _, _ = self.encoder(data, percent)
        global_graph = F.relu(self.fc1(global_graph))
        pred = self.fc2(global_graph)
        pred = pred.view(-1)
        return pred

    def unsup_loss(self, data, percent):
        if self.separate_encoder:
            pos_global_graph, sampling_subgraph, neg_global_graph, prob_loss = self.unsup_encoder(data, percent,
                                                                                                  eval=False)
        else:
            pos_global_graph, sampling_subgraph, neg_global_graph, prob_loss = self.encoder(data, percent, eval=False)

        # TODO:local MI loss: hn0? glabal_node?
        global_graph_embed = self.global_d(pos_global_graph)
        sub_graph_embed = self.local_d(sampling_subgraph)
        neg_global_graph_embed = self.global_d(neg_global_graph)

        # l_enc_glb_hn0 = self.local_d(h_n0)
        # sub_batch = torch.arange(global_graph_embed.size(0)).unsqueeze(1).repeat_interleave(self.num_subgraph, 1).view(-1)
        measure = 'JSD'
        loss_graph_subgraph = global_global_loss_(global_graph_embed, sub_graph_embed, neg_global_graph_embed, measure)

        if not self.use_prob_loss:
            prob_loss=0

        return loss_graph_subgraph + prob_loss

    def unsup_sup_loss(self, data, percent):
        global_graph, global_sub_graph, neg_global_sub_graph, prob_loss = self.encoder(data, percent, eval=False)
        global_graph_, global_sub_graph_, neg_global_sub_graph_, prob_loss_ = self.unsup_encoder(data, percent,
                                                                                                 eval=False)

        if not self.use_prob_loss:
            prob_loss, prob_loss_ = 0, 0

        global_graph_embed = self.ff1(global_graph)
        global_graph_embed_ = self.ff2(global_graph_)

        measure = 'JSD'
        loss = global_global_loss_(global_graph_embed, global_graph_embed_, data.batch, measure)
        return loss + prob_loss + prob_loss_
