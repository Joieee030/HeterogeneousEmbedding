import random
import pandas as pd


class GeneratorData(object):

    def __init__(self, node_path, edge_path, walk_length, num_per_vertex, output_file):
        self.nodes = pd.DataFrame(pd.read_excel(node_path))
        self.graph = pd.DataFrame(pd.read_excel(edge_path))
        self.walk_length = walk_length
        self.num_per_vertex = num_per_vertex
        self.output_file = output_file

    # 定义游走规则
    def random_walk(self, start_point, walk_length):
        df = self.graph

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

    def save_sentences(self):
        print("data for train generating...")
        file_out = open(self.output_file, 'w')
        df_node = self.nodes
        epoch = self.num_per_vertex
        start_nodes = set([i for i in range(df_node.shape[0]) if df_node.iloc[i]["类型"] == "paper"])
        for start_node in start_nodes:
            for _ in range(epoch):
                sentence = self.random_walk(start_node, walk_length=self.walk_length)
                for item in sentence:
                    file_out.write("%s " % item)
                file_out.write("\n")
        print("Done!")


if __name__ == '__main__':
    node_path = "../../data/node_attr.xlsx"
    edge_path = "../../data/relation.xlsx"
    walk_length = 100
    num_per_vertex = 5
    output_name = "./gen.txt"
    test = GeneratorData(node_path, edge_path, walk_length, num_per_vertex, output_name)
    test.save_sentences()
