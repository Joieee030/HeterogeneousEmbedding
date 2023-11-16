import torch
import torch.nn as nn
from torch_geometric.graphgym import optim
from deepwalk.mydeepwalk import MyDeepWalk
from parameters import args
from model.hierarchical_softmax import HierarchicalSoftmax


if __name__ == '__main__':
    node_path = args.node_path
    edge_path = args.edge_path
    window_size = 4
    walk_length = 100
    num_per_vertex = 5
    output_size = 100
    deepwalk = MyDeepWalk(node_path, edge_path, window_size, walk_length, num_per_vertex, output_size)
    word2vec_model = deepwalk.word2vec()

    # 定义输入大小和隐藏层大小
    input_size = 100  # 假设词向量维度为100
    hidden_size = 50

    model = HierarchicalSoftmax(word2vec_model, input_size, hidden_size)

    # 定义超参数
    learning_rate = 0.001
    num_epochs = 10

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        total_loss = 0.0
        for word in word2vec_model.wv.key_to_index:  # 遍历所有单词
            target_prob = 1.0  # 假设目标概率为1.0，这里可以根据实际情况调整

            # 前向传播、计算损失和反向传播
            loss = model.backward(word, target_prob)
            total_loss += loss.item()

            # 更新模型参数
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(word2vec_model.wv.key_to_index)}")

    print("训练完成！")

