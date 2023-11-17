import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.w_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self._init_emb()

    def _init_emb(self):
        ini_trange = 0.5 / self.emb_dimension
        self.w_embeddings.weight.data.uniform_(-ini_trange, ini_trange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_w, neg_v):
        emb_w = self.w_embeddings(torch.LongTensor(pos_w))  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        neg_emb_w = self.w_embeddings(torch.LongTensor(neg_w))
        emb_v = self.v_embeddings(torch.LongTensor(pos_v))
        neg_emb_v = self.v_embeddings(
            torch.LongTensor(neg_v))  # 转换后大小 [ negative_sampling_number * mini_batch_size * emb_dimension ]
        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(-1 * score)
        neg_score = torch.mul(neg_emb_w, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(neg_score)
        # L = log sigmoid (Xw.T * θv) + [log sigmoid (-Xw.T * θv)]
        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        return loss

    def save_embedding(self, id2word, file_name):
        embedding = self.w_embeddings.weight.data.numpy()
        file_out = open(file_name, 'w')
        file_out.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            file_out.write('%s %s\n' % (w, e))
