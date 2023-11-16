# from node import Node

class Node:
    def __init__(self, word=None, frequency=0):
        self.word = word
        self.frequency = frequency
        self.left = None
        self.right = None
        self.is_leaf()

    def is_leaf(self):
        return self.left is None and self.right is None


class HuffmanTree:

    def __init__(self, data):
        self.root = None
        self.data = data
        self.build_tree()

    def build_tree(self):
        nodes = [Node(value, frequency) for value, frequency in self.data.items()]
        while len(nodes) > 1:
            nodes = sorted(nodes, key=lambda x: x.frequency)

            left_child = nodes[0]
            right_child = nodes[1]

            parent_frequency = left_child.frequency + right_child.frequency
            parent_node = Node(None, parent_frequency)
            parent_node.left = left_child
            parent_node.right = right_child

            nodes = nodes[2:]
            nodes.append(parent_node)
        self.root = nodes[0]
        return nodes[0]

    def __getitem__(self, key):
        return key in self.data
