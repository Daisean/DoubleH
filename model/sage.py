import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair


class SAGENet(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(WeightedSAGEConv(in_size, hid_size, hid_size))
            elif i == num_layers - 1:
                self.layers.append(WeightedSAGEConv(hid_size, hid_size, out_size))
            else:
                self.layers.append(WeightedSAGEConv(hid_size, hid_size, hid_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
        return h

class WeightedSAGEConv(nn.Module):
    def __init__(self, in_size, hid_size, out_size, activation=F.relu, dropout=0.1):
        super().__init__()

        self.activation = activation
        self.Q = nn.Linear(in_size, hid_size)
        self.W = nn.Linear(in_size + hid_size, out_size)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0.)
        nn.init.constant_(self.W.bias, 0.)

    def forward(self, g, h):
        h_src, h_dst = expand_as_pair(h, g)
        with g.local_scope():
            g.srcdata['n'] = h_src

            g.edata['count'] = g.edata['count'].float()
            g.update_all(fn.u_mul_e('n', 'count', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('count', 'm'), fn.sum('m', 'w'))

            n = g.dstdata['n']
            w = g.dstdata['w'].unsqueeze(1).clamp(min=1)
            z = self.activation(self.W(self.dropout(torch.cat([n / w, h_dst], dim=1))))

            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm

            return z