import dgl
from model import gnns, doubleh, sage
from sampler import NeighborSampler, Sagesampler

class Model_handler():
    def __init__(self, args):
        self.config = args
        self.model_name = args.model
        if self.model_name in ['doubleh', 'gat', 'gin', 'gcn', 'sage', 'pinsage']:
            self.mode = 'homo'
        else:
            self.mode = 'hete'

    def get_model(self, input_size, args, extra_info):
        if self.model_name == 'sage':
            model = sage.SAGENet(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                                 num_layers=args.num_layers)
            sampler = Sagesampler(
                random_walk_length=1, num_random_walks=args.num_random_walks,
                num_neighbors=args.num_neighbors, num_layers=args.num_layers)

        elif args.model == 'pinsage':
            model = sage.SAGENet(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             num_layers=args.num_layers)
            sampler = Sagesampler(
                random_walk_length=args.random_walk_length, num_random_walks=args.num_random_walks,
                num_neighbors=args.num_neighbors, num_layers=args.num_layers)

        elif args.model == 'gcn':
            model = gnns.GCN(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             num_layers=args.num_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.random_walk_length)

        elif args.model == 'rgcn':
            model = gnns.RGCN(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             rel_names=extra_info, num_layers=args.num_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.random_walk_length)

        elif args.model == 'rgat':
            model = gnns.RGAT(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             rel_names=extra_info, num_layers=args.num_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.random_walk_length)

        elif args.model == 'gin':
            model = gnns.GIN(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             num_layers=args.num_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.random_walk_length)

        elif args.model == 'gat':
            model = gnns.GAT(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             num_layers=args.num_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.random_walk_length)

        elif args.model == 'doubleh':
            model = doubleh.GNet(in_size=input_size, hid_size=args.hidden_size, out_size=2,
                             num_layers=args.num_layers)
            sampler = NeighborSampler(
                random_walk_length=args.random_walk_length, num_random_walks=args.num_random_walks,
                num_neighbors=args.num_neighbors, num_layers=args.num_layers)
        else:
            model, sampler = None, None


        return model, sampler
