import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting random seed to facilitate reproduction
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

"""
0. embedding: Linear -> Relu
1. encoder: Encoder -> EncoderLayer*6: MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection*2;
2. pma: Encoder -> PoolingLayer*1: MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection*1;
3. prediction: Linear -> Relu -> Linear -> Sigmoid
"""


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


class QError(nn.Module):
    def __init__(self):
        super(QError, self).__init__()

    def forward(self, preds, targets, out="mean",
                # min_val=-16.997074, max_val=2.9952054,
                min_val=None, max_val=None):
        if min_val and max_val:
            preds = unnormalize_torch(preds, min_val, max_val)
            targets = unnormalize_torch(targets, min_val, max_val)
        else:
            preds = preds + 1e-8
            targets = targets + 1e-8

        x = preds / targets
        y = targets / preds
        qerror = torch.where(preds > targets, x, y)

        if out == "mean":
            return torch.mean(qerror)
        elif out == "raw":
            return qerror


# Find Q-Error
def q_error(actual, pred):
    epsilon = 1e-4
    q_e = 0

    for i in range(0, len(pred)):
        q_e += max((actual[i] + epsilon) / (pred[i] + epsilon), (pred[i] + epsilon) / (actual[i] + epsilon))

    return q_e / len(pred)


# To output median, 90th quantile and 95th quantile errors
def find_median_90_95(actual, pred):
    qe = []

    for i in range(0, len(pred)):
        actual_v = actual[i]
        pred_v = pred[i]
        qe.append(max((pred_v + 1e-4) / (actual_v + 1e-4), (actual_v + 1e-4) / (pred_v + 1e-4)))

    res = [np.median(qe), np.percentile(qe, 90), np.percentile(qe, 95)]

    return res


# Model Evaluation on test data
def eval_on_test_set(model, bs, device, len_test,
                     test_data, test_label, test_mask):
    model.eval()

    running_error_qe = 0
    num_batches = 0
    _pred = []
    _label = []

    for i in range(0, len_test, bs):
        minibatch_data = test_data[i:i + bs].to(device)
        minibatch_label = test_label[i:i + bs].numpy()
        minibatch_mask = test_mask[i:i + bs].to(device)

        # zw: torch.Size([20, 1, 1]) -> (20, 1)
        pred_rr = model(minibatch_data, minibatch_mask).detach().cpu().squeeze(dim=1).numpy()
        _pred.extend(pred_rr)
        _label.extend(minibatch_label)

        qe = q_error(minibatch_label, pred_rr)

        running_error_qe += qe

        num_batches += 1

    total_qe = running_error_qe / num_batches
    test_res_h = find_median_90_95(_label, _pred)

    print(f'Mean Test Q_error = {total_qe}\n')
    print('Median:')
    print(f'Median Test Q_error = {test_res_h[0]}\n')
    print('90 Percentile:')
    print(f'90th Test Q_error = {test_res_h[1]}\n')
    print('95 Percentile:')
    print(f'95th Test Q_error = {test_res_h[2]}\n')


# LIB consists of three parts: (1) feature extractor, (2) Encoder and (3) Prediction model.
# This notebook shows the design of the encoder and the prediction model.

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # torch.Size([20, 8, 36, 36])
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.4):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 4 = 3(q,k,v) + 1(fc).
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # torch.Size([20, 36]) -> torch.Size([20, 1, 1, 36])
        nbatches = query.size(0)
        # torch.Size([20, 36, 32]) -> torch.Size([20, 36, 8, 4]) -> torch.Size([20, 8, 36, 4])
        query, key, value = \
            [l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # torch.Size([20, 8, 36, 4]), torch.Size([20, 8, 36, 36])
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # torch.Size([20, 36, 32])
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        return self.linears[-1](x)


# FFN
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.4):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, y, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(y, x, x, mask))

        return self.sublayer[1](x, self.feed_forward)


# Pooling by attention
class PoolingLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(PoolingLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, y, x, mask):
        x = self.self_attn(y, x, x, mask)
        return self.sublayer(x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, y, x, mask):
        for layer in self.layers:
            x = layer(y, x, mask)
        return self.norm(x)


def make_model(d_model, N, d_ff, n_head, dropout=0.4):
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # EncoderLayer(size, self_attn, feed_forward, dropout)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    # PoolingLayer(size, self_attn, feed_forward, dropout)
    pooling_model = Encoder(PoolingLayer(d_model, c(attn), c(ff), dropout), 1)

    return model, pooling_model


# Completed LIB
class self_attn_model(nn.Module):
    def __init__(self, encoder, pooling_model, ini_feats, encode_feats, hidden_dim):
        super(self_attn_model, self).__init__()

        self.linear1 = nn.Linear(ini_feats, encode_feats, bias=True)

        self.encoder = encoder
        self.pma = pooling_model
        self.S = nn.Parameter(torch.Tensor(1, 1, encode_feats))
        nn.init.xavier_uniform_(self.S)

        self.output1 = nn.Linear(encode_feats, hidden_dim, bias=True)
        self.output2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, batch_samples, batch_mask):
        batch_samples = F.relu(self.linear1(batch_samples))  # zw: torch.Size([20, 36, 12]) -> torch.Size([20, 36, 32])
        attn_output = self.encoder(batch_samples, batch_samples, batch_mask)  # torch.Size([20, 36, 32])

        # Z: self.S.repeat(attn_output.size(0), 1, 1)
        attn_output = self.pma(self.S.repeat(attn_output.size(0), 1, 1),
                               attn_output, batch_mask)  # torch.Size([20, 1, 32])
        hidden_rep = F.relu(self.output1(attn_output))
        out = torch.sigmoid(self.output2(hidden_rep))

        return out
