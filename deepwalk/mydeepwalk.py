import random
import pandas as pd
from gensim.models import word2vec


class MyDeepWalk(object):

    def __init__(self, node_path, edge_path, window_size, walk_length, num_per_vertex, output_size):
        self.nodes = pd.DataFrame(pd.read_excel(node_path))
        self.graph = pd.DataFrame(pd.read_excel(edge_path))
        self.window_size = window_size
        self.walk_length = walk_length
        self.num_per_vertex = num_per_vertex
        self.output_size = output_size

    # 定义游走规则
    def random_walk(self, start_points, walk_length):
        df = self.graph

        for start_point in start_points:
            walk_sequence = [start_point]
            current_point = start_point
            step = 0

            while step < walk_length:
                # 从数据框中筛选与当前点相关的行
                filtered_df = df[df['点1'] == current_point]

                if len(filtered_df) == 0:
                    break

                # 随机选择下一个点
                next_index = random.randint(0, len(filtered_df) - 1)
                next_point = filtered_df.iloc[next_index]['点2']

                # 添加到游走序列中
                walk_sequence.append(next_point)
                current_point = next_point
                step += 1

            return walk_sequence

    def word2vec(self):
        print("skip-gram start...")
        df_node = self.nodes
        epoch = self.num_per_vertex
        start_nodes = set([i for i in range(df_node.shape[0]) if df_node.iloc[i]["类型"] == "paper"])
        sentences = []

        for _ in range(epoch):
            sentences.append(self.random_walk(start_nodes, walk_length=self.walk_length))
        # 使用skip-gram模型训练
        model = word2vec.Word2Vec(sentences, vector_size=self.output_size, window=self.window_size, min_count=0, sg=1,
                                  workers=4)
        print("Done!")
        return model

    def test_word2vec(self):
        df_node = self.nodes
        start_nodes = set([i for i in range(df_node.shape[0]) if df_node.iloc[i]["类型"] == "paper"])
        print("nodes: ", start_nodes)
        model = self.word2vec()
        # print(model.wv.key_to_index)
        for item in start_nodes:
            # print(model.wv.most_similar(positive=[item], topn=3))  # 相似度前三
            # print("representation of paper_{} is:\n {} ".format(item, model.wv[item]))  # 节点的嵌入表示
            print("vector of paper_{} max: {}, min: {}".format(item, model.wv[item].max(), model.wv[item].min()))
