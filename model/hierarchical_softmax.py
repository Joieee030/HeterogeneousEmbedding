import torch
import torch.nn as nn
from huffman.tree import HuffmanTree


class HierarchicalSoftmax(nn.Module):
    def __init__(self, word2vec_model, input_size, hidden_size):
        super(HierarchicalSoftmax, self).__init__()
        self.word2vec_model = word2vec_model
        self.huffman_tree = HuffmanTree(word2vec_model.wv.key_to_index)
        self.word_vector = word2vec_model.wv.key_to_index
        self.input_size = input_size

        # 初始化线性层
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)  # 输出大小为2，节点概率为二分类
        self.sigmoid = nn.Sigmoid()

        self.parameters = [self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias]
        for param in self.parameters:
            param.requires_grad = True

    def forward(self, input_word):
        current_node = self.huffman_tree.root  # 从根节点开始
        log_prob_sum = self.traverse_tree(input_word, current_node)  # 调用递归函数
        return log_prob_sum

    def traverse_tree(self, input_word, current_node):
        if current_node.is_leaf:  # 到达叶子节点，返回0作为结束条件
            return 0.0

        word_vector = torch.tensor(self.word2vec_model.wv[input_word])  # 获取输入单词的向量表示
        output = self.linear2(self.sigmoid(self.linear1(word_vector)))  # 计算当前节点的输出
        current_node_index = 0 if current_node.left_child else 1  # 确定当前节点的索引
        log_prob = torch.log(output[0, current_node_index])  # 取对数计算概率

        parent_node = current_node.parent
        log_prob_sum = log_prob + self.traverse_tree(input_word, parent_node)  # 递归调用继续向上遍历

        return log_prob_sum

    def my_loss(self, word, target_prob):
        pred_prob = torch.tensor(self.forward(word))  # 前向传播，计算预测概率

        loss = -torch.log(pred_prob) * target_prob  # 计算负对数损失
        return loss

    def backward(self, word, target_prob):
        self.zero_grad()

        loss = self.my_loss(word, target_prob)
        loss.backward()

        # 手动更新非叶子节点的向量表示
        with torch.no_grad():
            for param in self.parameters():
                if len(param.shape) == 2:
                    param.grad -= self.learning_rate * \
                                  self.sigmoid(self.word_vector[word]) * torch.mm(
                        torch.transpose(self.linear1.weight, 0, 1), param)

        return loss
