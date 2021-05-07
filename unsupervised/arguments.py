import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Gragh Enhance Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset', default='PTC_MR')
    parser.add_argument('--gpu',default=0,help='The number of gpu you used')

    parser.add_argument('--mode', dest='mode', type=str, default='TS',
                        help='MH for multi-head generation, TS for tree-spilit generation')
    parser.add_argument('--times', dest='times', type=int, default=2,
                        help='Number of partition times')
    parser.add_argument('--proloss', type=bool, default=False)
    parser.add_argument('--percent', dest='percent', type=float, default=0,
                        help='0 for end2end learning, [0,1] for specific subgraph percent')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=4,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128,
                        help='')

    return parser.parse_args()