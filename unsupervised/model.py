from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool, SAGPooling, GCNConv
from tqdm import tqdm
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import losses as losses
import random
from info_nce import InfoNCE


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class cat_mi(nn.Module):
    def __init__(self, embedded_dim):
        super(cat_mi, self).__init__()
        self.embedded_dim = embedded_dim
        self.fc = nn.Sequential(nn.Linear(2 * self.embedded_dim, 32),
                                nn.ReLU(True),
                                nn.Linear(32, 1))

    def forward(self, hn0, global_node):
        cat = torch.cat([hn0, global_node], 1)
        return self.fc(cat)


class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, times, mode):
        super(Encoder, self).__init__()

        self.mode = mode
        self.nce = InfoNCE()
        self.num_gc_layers = num_gc_layers

        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()

        self.ini_embed = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.layer_conv = nn.Conv2d(num_gc_layers, 1, (1, 1))

        self.mlp_before_readout = FF(dim)
        self.num_subgraph = 2 ** times

        self.subgraph_conv = nn.Conv2d(self.num_subgraph, 1, (1, 1))

        self.partition_list = nn.ModuleList()
        self.times = times
        for i in range(self.num_subgraph - 1):
            self.partition_list.append(nn.Sequential(nn.Linear(dim, 2), nn.Softmax(dim=-1)))

        self.mha_trans = nn.Parameter(torch.Tensor(self.num_subgraph, dim, 2), requires_grad=True)
        # for param in self.partition.parameters():
        #     nn.init.xavier_uniform_(param)

        self.dim = dim

        for i in range(num_gc_layers):
            mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(mlp)

            bn = torch.nn.BatchNorm1d(dim)

            self.conv_list.append(conv)
            self.bn_list.append(bn)

    def forward(self, x, edge_index, batch, percent):
        node_embed_list = []
        prob_loss = 0
        x = self.ini_embed(x)
        neg_hn0 = self.create_neg_n0(x, batch)
        x = torch.cat([x, neg_hn0], dim=0)
        edge_index_ = torch.cat([edge_index, edge_index + edge_index[0, -1] + 1], dim=-1)
        for i in range(self.num_gc_layers):
            # activation function: per layer or final layer?
            x = F.relu(self.conv_list[i](x, edge_index_))
            x = self.bn_list[i](x)
            node_embed_list.append(x)

        layer_nodes_embed = torch.stack(node_embed_list, dim=1).unsqueeze(-1)

        global_node_embed = self.layer_conv(layer_nodes_embed).squeeze().view(2, -1, self.dim)
        pos_global_node, neg_global_node = global_node_embed[0], global_node_embed[1]
        neg_global_graph = global_add_pool(neg_global_node, batch)

        if percent:
            mask = self.sampling_subgraph(percent, batch)
            sampling_subgraph_list = []
            for i in range(self.num_sub_graph):
                sampling_subgraph_list.append(global_add_pool(pos_global_node[mask[i, :]], batch=batch[mask[i, :]]))
            sampling_subgraph_ = torch.stack(sampling_subgraph_list, dim=1).unsqueeze(1)
            sampling_subgraph = self.subgraph_conv(sampling_subgraph_).squeeze()
        elif self.mode == 'MH':
            sampling_subgraph, assignment_prob = self.multi_head_subgraph_generation(pos_global_node, batch)
            prob_loss = -torch.sum(torch.abs(assignment_prob[:, :, 0] - assignment_prob[:, :, 1]))
        elif self.mode == 'TS':
            sampling_subgraph_list, assignment_prob = self.generate_subgraphs(pos_global_node, batch, self.times)
            prob_loss = -torch.sum(torch.abs(assignment_prob[:, :, 0] - assignment_prob[:, :, 1]))
            sampling_subgraph_ = torch.stack(sampling_subgraph_list, dim=1).unsqueeze(-1)
            sampling_subgraph = self.subgraph_conv(sampling_subgraph_).squeeze()
        else:
            assert False, 'wrong parameter for subgraphs'
        pos_global_graph = global_add_pool(pos_global_node, batch)

        info_nce = self.nce_out(pos_global_node, pos_global_graph, neg_global_graph, batch)
        return pos_global_graph, sampling_subgraph, neg_global_graph, prob_loss, info_nce

    def nce_out(self, pos_node, pos_graph, neg_graph, batch):
        out = 0
        for i in range(batch[-1] + 1):
            all_ = 0
            for j in pos_node[batch == i]:
                all_ += self.nce(j.unsqueeze(0), pos_graph[0].unsqueeze(0),pos_graph )

            out += all_ / (batch == i).sum()
        return out / (batch[-1] + 1)

    def sampling_subgraph(self, percent, batch):
        mask = torch.stack([torch.zeros_like(batch) for i in range(self.num_subgraph)],
                           dim=0)  # num_sub_graph x num_total_nodes

        for _ in range(int(self.num_subgraph)):
            for k, p in enumerate(percent):
                last_node_num = 0
                for i in range(batch[-1] + 1):
                    node_num = batch[batch == i].size(0)
                    sample_node_num = int(np.ceil(node_num * p))
                    idx = np.random.choice(node_num, sample_node_num)
                    mask[_ + k, idx + last_node_num] = 1
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
            subgraph_list.append(global_add_pool(node_embed_list[i], batch))
        assign_prob = torch.stack(assign_prob_list, dim=0)
        return subgraph_list, assign_prob

    def multi_head_subgraph_generation(self, node_embeddings, batch):
        P = torch.softmax(torch.matmul(node_embeddings, self.mha_trans), dim=-1)
        mask = (P >= 0.5)[:, :, 0].long()
        mask = torch.stack([mask for i in range(self.dim)], dim=-1)
        subgraph_node_embed = (node_embeddings * mask).permute(1, 0, 2).unsqueeze(-1)
        subgraph_embed = self.subgraph_conv(subgraph_node_embed).squeeze()
        return global_add_pool(subgraph_embed, batch), P

    def create_neg_n0(self, h_n0, batch):
        neg_hn0 = []
        for i in range(int(batch[-1] + 1)):
            mask = batch == i
            h = h_n0[mask, :]
            idx = np.random.choice(h.size(0), h.size(0))
            neg_hn0.append(h[idx])
        return torch.cat(neg_hn0, dim=0)


class GraphEnhance(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, mode, alpha=0.5, beta=1., gamma=.1, times=1):
        super(GraphEnhance, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.embedding_dim = hidden_dim
        self.times = times

        self.encoder = Encoder(num_features, self.embedding_dim, num_gc_layers, self.times, mode)
        self.node_node_mi = cat_mi(self.embedding_dim)
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, percent):
        # batch_size = data.num_graphs

        pos_global_graph, sampling_subgraph, neg_global_graph, prob_loss, infonce = self.encoder(x,
                                                                                                 edge_index,
                                                                                                 batch,
                                                                                                 percent)
        # =========================================
        # sub_batch = torch.arange(pos_global_graph.size(0)).unsqueeze(1).repeat_interleave(2 ** self.times, 1).view(
        #     -1)
        # local_loss = losses.local_global_loss_(pos_global_graph, sampling_subgraph_.view(-1, self.embedding_dim),
        #                                        sub_batch)
        # =========================================

        loss_graph_subgraph, E_neg1, E_neg2, E_pos = losses.global_global_loss_(pos_global_graph, sampling_subgraph,
                                                                                neg_global_graph,
                                                                                measure='JSD')

        return loss_graph_subgraph, prob_loss, E_neg1, E_neg2, E_pos, infonce
