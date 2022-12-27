import numpy as np
import dgl
import dgl.backend as F
from dgl.dataloading import BlockSampler
from dgl.sampling.randomwalks import random_walk
import torch

class NeighborSampler(BlockSampler):
    def __init__(self, random_walk_length, num_random_walks, num_neighbors, num_layers, termination_prob=0.5, num_traversals=1):
        super().__init__()
        
        self.random_walk_length = random_walk_length
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals
        self.num_layers = num_layers

        restart_prob = np.zeros(random_walk_length * num_traversals)
        restart_prob[random_walk_length::random_walk_length] = termination_prob
        self.restart_prob = F.tensor(restart_prob, dtype=F.float32)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for _ in range(self.num_layers):
            frontier = self.sample_neighbors(g, seed_nodes)
            block = dgl.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

    def sample_neighbors(self, g, seed_nodes):
        seed_nodes = dgl.utils.prepare_tensor(g, seed_nodes, 'seed_nodes')
        self.restart_prob = F.copy_to(self.restart_prob, F.context(seed_nodes))
        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, eids, types = random_walk(
            g, seed_nodes, length=self.random_walk_length, restart_prob=self.restart_prob, return_eids=True)

        src_1 = F.reshape(paths[:, self.random_walk_length-1::self.random_walk_length], (-1,))
        src_2 = F.reshape(paths[:, self.random_walk_length::self.random_walk_length], (-1,))

        src = torch.cat([src_1, src_2], dim=0)
        dst = torch.cat([seed_nodes, seed_nodes], dim=0)

        mask_1 = (src_1 != -1)
        mask_2 = (src_2 != -1)
        mask = (src != -1)
        src = src[mask]
        dst = dst[mask]
        eids = eids[mask_2]

        neighbor_graph = dgl.graph((src, dst))
        etypes = g.edata[dgl.ETYPE][eids]

        neighbor_graph.edata['etype'] = torch.cat([etypes[:, 0], etypes[:, 1]], dim=0)
        neighbor_graph.edata['hete'] = torch.cat([torch.ones(etypes.shape[0]), torch.zeros(etypes.shape[0])])
        neighbor_graph.edata['homo'] = torch.cat([torch.zeros(etypes.shape[0]), torch.ones(etypes.shape[0])])
        neighbor_graph = dgl.to_simple(neighbor_graph, return_counts='count', copy_edata=True)

        return neighbor_graph


class Sagesampler(BlockSampler):
    def __init__(self, random_walk_length, num_random_walks, num_neighbors, num_layers, termination_prob=0.5, num_traversals=1):
        super().__init__()

        self.random_walk_length = random_walk_length
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals
        self.num_layers = num_layers

        restart_prob = np.zeros(random_walk_length * num_traversals)
        restart_prob[random_walk_length::random_walk_length] = termination_prob
        self.restart_prob = F.tensor(restart_prob, dtype=F.float32)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for i in range(self.num_layers):
            frontier = self.sample_neighbors(g, seed_nodes)
            block = dgl.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

    def sample_neighbors(self, g, seed_nodes):
        seed_nodes = dgl.utils.prepare_tensor(g, seed_nodes, 'seed_nodes')
        self.restart_prob = F.copy_to(self.restart_prob, F.context(seed_nodes))

        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, eids, types = random_walk(
            g, seed_nodes, length=self.random_walk_length, restart_prob=self.restart_prob, return_eids=True)

        src = F.reshape(paths[:, self.random_walk_length::self.random_walk_length], (-1,))
        dst = seed_nodes

        mask = (src != -1)
        eids = eids[mask]
        src = src[mask]
        dst = dst[mask]
        
        neighbor_graph = dgl.graph((src, dst))
        etypes = g.edata[dgl.ETYPE][eids]

        neighbor_graph.edata['first_etype'] = etypes[:, 0]
        if self.random_walk_length == 2:
            neighbor_graph.edata['second_etype'] = etypes[:, 1]
        neighbor_graph = dgl.to_simple(neighbor_graph, return_counts='count', copy_edata=True)

        return neighbor_graph