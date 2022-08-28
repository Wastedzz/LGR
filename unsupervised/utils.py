import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.utils import degree, remove_self_loops


def move_to(data, device=None):
    batch = data.batch.to(device)
    edge_idx = data.edge_index.to(device)
    if data.x is None:
        node_attr = degree(edge_idx[0], batch.shape[0]).view(-1, 1).to(device)
        # node_attr = torch.ones((batch.shape[0], 1)).to(device)
    else:
        node_attr = data.x.to(device)
    if data.edge_attr is None:
        edge_attr = data.edge_attr
    else:
        edge_attr = data.edge_attr.to(device)
    if data.y is None:
        y = data.y
    else:
        y = data.y.to(device)
    edge_idx, edge_attr = remove_self_loops(edge_idx, edge_attr)
    return node_attr, edge_attr, edge_idx, batch, y


def get_embeddings(Encoder, loader, percent, device=None):
    ret = []
    label = []
    with torch.no_grad():
        for data in loader:
            node_attr, edge_attr, edge_idx, batch, y = move_to(data, device)
            pos_global_graph, _, _, _,_ = Encoder(node_attr, edge_idx, batch, percent)
            # _, pos_global_graph, _ = Encoder(node_attr, edge_idx, batch, percent)
            ret.append(pos_global_graph.cpu().numpy())
            label.append(y.cpu().numpy())
    ret = np.concatenate(ret, 0)
    label = np.concatenate(label, 0)
    return ret, label
