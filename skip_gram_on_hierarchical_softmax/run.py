"""
测试word_embedding效果
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

f = open('word_embedding.txt')
f.readline()
all_embeddings = []
all_words = []
word2id = dict()
for i, line in enumerate(f):
    line = line.strip().split(' ')
    word = line[0]
    embedding = [float(x) for x in line[1:]]
    assert len(embedding) == 100
    all_embeddings.append(embedding)
    all_words.append(word)
    word2id[word] = i
all_embeddings = np.array(all_embeddings)
print("paper dict: \n", word2id)

# df_node = pd.DataFrame(pd.read_excel("../data/node_attr.xlsx"))
# papers = set([i for i in range(df_node.shape[0]) if df_node.iloc[i]["类型"] == "paper"])

while 1:
    word = input('paper: ')
    try:
        wid = word2id[word]
    except:
        print('Cannot find this paper')
        continue
    embedding = all_embeddings[wid:wid + 1]
    d = cosine_similarity(embedding, all_embeddings)[0]
    d = zip(all_words, d)
    d = sorted(d, key=lambda x: x[1], reverse=True)

    for w in d[1:]:
        print(w)
