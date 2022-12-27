from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.nn as dglnn



class RGCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, rel_names, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for i in range(0, num_layers):
            if num_layers == 1:
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_size, out_size) for rel in rel_names}, aggregate='sum'))
            else:
                if i == 0:
                    self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_size, hid_size) for rel in rel_names}, aggregate='sum'))
                elif i == num_layers - 1:
                    self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_size, out_size) for rel in rel_names}, aggregate='sum'))
                else:
                    self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_size, hid_size) for rel in rel_names}, aggregate='sum'))

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = {k: self.dropout(F.relu(v)) for k, v in h.items()}

        return h['user']

class RGAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, rel_names, num_layers) -> None:
        super().__init__()
        self.head = 2
        self.hidden = hid_size
        self.linlayers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for i in range(0, num_layers):
            if num_layers == 1:
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(in_size, out_size, num_heads=self.head) for rel in rel_names}, aggregate='sum'))
                self.linlayers.append(nn.Linear(2 * out_size, out_size))
            else:
                if i == 0:
                    self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(in_size, hid_size, num_heads=self.head) for rel in rel_names}, aggregate='sum'))
                    self.linlayers.append(nn.Linear(2 * hid_size, hid_size))
                elif i == num_layers - 1:
                    self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(hid_size, out_size, num_heads=self.head) for rel in rel_names}, aggregate='sum'))
                    self.linlayers.append(nn.Linear(2 * out_size, out_size))
                else:
                    self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(hid_size, hid_size, num_heads=self.head) for rel in rel_names}, aggregate='sum'))
                    self.linlayers.append(nn.Linear(2 * hid_size, hid_size))

    def forward(self, blocks, h):
        for i, (layer, lin, block) in enumerate(zip(self.layers, self.linlayers, blocks)):
            h = layer(block, h)
            h = {k: lin(v.view(-1, 2 * v.size(-1))) for k, v in h.items()}
            if i != len(self.layers) - 1:
                h = {k: self.dropout(F.relu(v)) for k, v in h.items()}

        return h['user']


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for i in range(0, num_layers):
            if num_layers == 1:
                self.layers.append(dglnn.GraphConv(in_size, out_size))
            else:
                if i == 0:
                    self.layers.append(dglnn.GraphConv(in_size, hid_size))
                elif i == num_layers - 1:
                    self.layers.append(dglnn.GraphConv(hid_size, out_size))
                else:
                    self.layers.append(dglnn.GraphConv(hid_size, hid_size))

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GIN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for i in range(0, num_layers):
            if num_layers == 1:
                self.layers.append(dglnn.GINConv(nn.Linear(in_size, out_size), 'max'))
            else:
                if i == 0:
                    self.layers.append(dglnn.GINConv(nn.Linear(in_size, hid_size), 'max'))
                elif i == num_layers - 1:
                    self.layers.append(dglnn.GINConv(nn.Linear(hid_size, out_size), 'max'))
                else:
                    self.layers.append(dglnn.GINConv(nn.Linear(hid_size, hid_size), 'max'))

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.head = 2
        self.hidden = hid_size
        self.linlayers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for i in range(0, num_layers):
            if num_layers == 1:
                self.layers.append(dglnn.GATConv(in_size, out_size, num_heads=self.head))
                self.linlayers.append(nn.Linear(2 * out_size, out_size))
            else:
                if i == 0:
                    self.layers.append(dglnn.GATConv(in_size, hid_size, num_heads=self.head))
                    self.linlayers.append(nn.Linear(2 * hid_size, hid_size))
                elif i == num_layers - 1:
                    self.layers.append(dglnn.GATConv(hid_size, out_size, num_heads=self.head))
                    self.linlayers.append(nn.Linear(2 * out_size, out_size))
                else:
                    self.layers.append(dglnn.GATConv(hid_size, hid_size, num_heads=self.head))
                    self.linlayers.append(nn.Linear(2 * hid_size, hid_size))

    def forward(self, blocks, h):
        for i, (layer, lin, block) in enumerate(zip(self.layers, self.linlayers, blocks)):
            tmp = layer(block, h)
            h = lin(tmp.view(-1, self.head * tmp.size(-1)))
            if i != len(self.layers) - 1:
                h = self.dropout(F.relu(h))

        return h