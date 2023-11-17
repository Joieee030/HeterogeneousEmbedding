import argparse


parser = argparse.ArgumentParser(description='PyTorch HeterogeneousEmbedding Model')
parser.add_argument('--edge_path', type=str, default='./data/relation.xlsx', help='edge path')
parser.add_argument('--node_path', type=str, default='./data/node_attr.xlsx', help='node path')
args = parser.parse_args()
