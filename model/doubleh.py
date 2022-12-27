import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair


class GNet(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.append(SAGEConv(in_size, hid_size, out_size))
            else:
                if i == 0:
                    self.layers.append(SAGEConv(in_size, hid_size, hid_size))
                elif i == num_layers - 1:
                    self.layers.append(SAGEConv(hid_size, hid_size, out_size))
                else:
                    self.layers.append(SAGEConv(hid_size, hid_size, hid_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
        return h


class SAGEConv(nn.Module):
    def __init__(self, in_size, hid_size, out_size, activation=F.relu, dropout=0.2):
        super().__init__()

        self.activation = activation
        self.text_proj = dglnn.linear.TypedLinear(in_size, hid_size, 4, regularizer='basis', num_bases=3)
        self.user_proj = dglnn.linear.TypedLinear(in_size, hid_size, 4, regularizer='basis', num_bases=3)
        self.dst_proj = dglnn.linear.TypedLinear(in_size, in_size, 4, regularizer='basis', num_bases=3)
        self.linear = nn.Linear(in_size + hid_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)

    def aggregator(self, edges):
        idx = 0
        idx_hete = []
        idx_homo = []
        # for i in edges.data['first_etype']:
        #     if i > 0.0:
        #         idx2.append(idx)
        #         idx += 1
        for i in range(edges.data['hete'].shape[0]):
            if edges.data['hete'][i] > 0.0:
                idx_hete.append(i)
            else:
                idx_homo.append(i)
                # idx += 1

        hete_feat = self.activation(self.text_proj(self.dropout(self.layer_norm(edges.src['n'][idx_hete])), edges.data['etype'][idx_hete]))
        homo_feat = self.activation(self.user_proj(self.dropout(self.layer_norm(edges.src['n'][idx_homo])), edges.data['etype'][idx_homo]))
        homo_empty = torch.zeros_like(homo_feat)
        hete_empty = torch.zeros_like(hete_feat)
        # feat = self.activation(self.dst_proj(self.dropout(self.layer_norm(edges.dst['n'])), edges.data['etype']))
        dst = self.activation(self.dst_proj(self.dropout(self.layer_norm(edges.dst['n'])), edges.data['etype']))
        # feat = self.activation(self.user_proj(self.dropout(self.layer_norm(edges.src['n'])), edges.data['etype']))
        feat = torch.cat([hete_feat, homo_feat], dim=0)
        feat = torch.cat([feat, dst], dim=-1)
        # m = feat * edges.data['count'].unsqueeze(-1)  # 这个效果不如下面那个好
        m = feat
        return {'m': m}
        #only user2


    def forward(self, g, h):
        h_src, h_dst = expand_as_pair(h, g)

        with g.local_scope():
            g.srcdata['n'] = h_src
            g.dstdata['n'] = h_dst

            g.edata['count'] = g.edata['count'].float()
            g.update_all(self.aggregator, fn.sum('m', 'n'))
            # g.update_all(fn.copy_e('count', 'm'), fn.sum('m', 'w'))

            n = g.dstdata['n']
            # w = g.dstdata['w'].unsqueeze(1).clamp(min=1)
            # z = self.activation(self.W(self.dropout(torch.cat([n / w, h_dst], dim=1))))
            # z = self.activation(self.linear(torch.cat([n, h_dst], dim=1)))
            z = self.activation(self.linear(n))

            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm


            return z
