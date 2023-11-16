from sklearn.metrics.pairwise import cosine_similarity
import numpy
from deepwalk.mydeepwalk import MyDeepWalk
from parameters import args


if __name__ == '__main__':
    node_path = args.node_path
    edge_path = args.edge_path
    window_size = 4
    walk_length = 100
    num_per_vertex = 5
    output_size = 100
    deepwalk = MyDeepWalk(node_path, edge_path, window_size, walk_length, num_per_vertex, output_size)
    # word2vec_model = deepwalk.word2vec()
    deepwalk.get_top_n(3)

