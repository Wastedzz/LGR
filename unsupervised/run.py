from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json, os
import numpy as np
import os.path as osp
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from arguments import arg_parse
from model import GraphEnhance
from utils import get_embeddings, move_to
from evaluate_embedding import evaluate_embedding
from torch.utils.tensorboard import SummaryWriter
import pprint as pp
import datetime
from tool import pd_toExcel


def compute(data_):
    data = data_.copy()
    data.remove(np.min(data))
    data.remove(np.max(data))
    r = np.mean(data)
    d = r - np.min(data)
    return (round(r * 100., 6), round(d * 100., 6)), (
        round(np.mean(data_) * 100., 6), round((np.mean(data_) - np.min(data_)) * 100., 6))


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

if __name__ == "__main__":
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # for args.mode in ['TS', 'MH']:
    #     for args.proloss in [True, False]:
    for args.times in [2]:
        pp.pprint(vars(args))
        epochs = 200
        log_interval = 5
        batch_size = 128

        lr = args.lr
        DS = args.DS
        mode = args.mode
        percent = args.percent
        times = args.times

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.proloss:
            pro_loss = 'withProLoss'
        else:
            pro_loss = 'NoProLoss'

        best_list = []
        for i in range(7):
            accuracies = {'logreg': [], 'svc': [], 'linearsvc': [], 'randomforest': []}

            dataset = TUDataset('../../ds', name=DS).shuffle()
            num_features = max(dataset.num_features, 1)
            dataloader = DataLoader(dataset, batch_size=batch_size)

            log_dir = os.path.join(f'log{args.hidden_dim}', DS, '{}'.format(mode),
                                   pro_loss,
                                   'times-{}'.format(times),
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            writer = SummaryWriter(log_dir)

            # print('================')
            # print('lr: {}'.format(lr))
            # print('num_features: {}'.format(num_features))
            # print('hidden_dim: {}'.format(args.hidden_dim))
            # print('num_gc_layers: {}'.format(args.num_gc_layers))
            # print('================')

            device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
            model = GraphEnhance(num_features, args.hidden_dim, args.num_gc_layers, mode, times=times).to(
                device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # print('\nTraining in {}-th loop'.format(i + 1))
            # try:
            #     with tqdm(range(1, epochs + 1), desc='{}-th loop'.format(i + 1)) as tadm_range:
            #
            nce_data = {'epoch': [], 'nce': []}
            for epoch in tqdm(range(1, epochs + 1), desc='{}-th loop'.format(i + 1), ncols=80):
                loss_all = 0
                E_pos_all = 0
                E_neg1_all = 0
                E_neg2_all = 0
                all_nce = 0
                model.train()
                for data in dataloader:
                    node_attr, edge_attr, edge_idx, batch, y = move_to(data, device)
                    optimizer.zero_grad()
                    loss, ProLoss, E_neg1, E_neg2, E_pos, nce = model(node_attr, edge_idx, batch, percent=percent)
                    all_nce += nce
                    if args.proloss:
                        loss = loss + ProLoss
                    loss_all += loss.item() * data.num_graphs
                    E_pos_all += E_pos
                    E_neg1_all += E_neg1
                    E_neg2_all += E_neg2
                    loss.backward()
                    optimizer.step()
                # print()
                # print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))
                # writer.add_scalar('GraphEnhance Pos Loss', -E_pos_all / len(dataloader), epoch)
                # writer.add_scalar('GraphEnhance Neg Loss1', E_neg1_all / len(dataloader), epoch)
                # writer.add_scalar('GraphEnhance Neg Loss2', E_neg2_all / len(dataloader), epoch)
                writer.add_scalar('nce', all_nce / len(dataloader), epoch)
                writer.add_scalar('GraphEnhance Loss', loss_all / len(dataloader), epoch)
                nce_data['epoch'].append(epoch)
                nce_data['nce'].append(all_nce.item() / len(dataloader))
                # if epoch % log_interval == 0:
                #     model.eval()
                #     emb, y = get_embeddings(model.encoder, dataloader, percent, device)
                #     res = evaluate_embedding(emb, y)
                #     accuracies['logreg'].append(res[0])
                #     accuracies['svc'].append(res[1])
                #     accuracies['linearsvc'].append(res[2])
                #     acc = [accuracies['logreg'], accuracies['svc'], accuracies['linearsvc']]
                #     acc_ = np.array(acc)
                #     best = acc_.max()
                #     # print('_'*100)
                #     # print(best)
                #     # print('_'*100)
                #     # pp.pprint(accuracies)
                #     # print(accuracies)
                #
                #     writer.add_scalar('logreg', res[0], epoch)
                #     writer.add_scalar('svc', res[1], epoch)
                #     writer.add_scalar('linearsvc', res[2], epoch)
                #     writer.add_scalar('Best', best, epoch)
                # writer.add_scalar('randomforest', res[3], epoch

                # =====================================
                del data
                torch.cuda.empty_cache()
                # =====================================

            # print(f'{i + 1}-th loop: best {round(best * 100., 6)}')
            # best_list.append(best)
            #     if i + 1 > 2:
            #         print(f'Best is {compute(best_list)} in {i + 1} loops')
            #
            # print('=' * 100)
            # print('num_subgraph:{}'.format(2 ** times), compute(best_list))
            # print('=' * 100)
            pd_toExcel(nce_data, 'info_nce_{}.xlsx'.format(10+i))

        p1 = os.path.join(f'textlog{args.hidden_dim}')
        if not os.path.exists(p1):
            os.makedirs(p1)
        with open(f'textlog{args.hidden_dim}/uns' + '-' + mode + '-' + DS + '-' + pro_loss + '.log', 'a+') as f:
            f.write(
                'num_subgraph:{}, minus max_min: {}, direct avg:{}\n'.format(2 ** times, compute(best_list)[0],
                                                                             compute(best_list)[1]))
