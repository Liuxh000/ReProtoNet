import numpy as np
import torch
from torch import nn
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.Q = nn.Conv2d(in_channels=d_model, out_channels=h*d_k, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(in_channels=d_model, out_channels=h*d_k, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(in_channels=d_model, out_channels=h*d_v, kernel_size=1, stride=1, padding=0)
        self.Conv_out = nn.Conv2d(in_channels=h*d_v, out_channels=d_model, kernel_size=1, stride=1, padding=0)


        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h



    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        qn, qc, qh, qw = queries.shape
        kn, kc, kh, kw = queries.shape
        vn, vc, vh, vw = queries.shape

        q = self.Q(queries).view(qn, self.h, qc, qh*qw).permute(0, 1, 3, 2)  # (5, 8, 9, 512)
        k = self.K(keys).view(kn, self.h, kc, kh*kw)  # (5, 8, 512, 9)
        v = self.V(values).view(vn, self.h, vc, vh*vw)  # (5, 8, 512, 9)

        att = torch.matmul(q, k) / np.sqrt(self.d_k/self.h)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1).permute(0, 1, 3, 2)


        out = torch.matmul(v, att).view(vn, self.h*vc, vh, vw)
        out = self.Conv_out(out)  # (b_s, nq, d_model)

        return out