"""
Created on Wednesday, April 27, 2022

@author: Guangxing Han
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer as https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_conv = nn.Conv2d(in_features, out_features, 1, padding=0, bias=False)
        self.norm = nn.LayerNorm([out_features, 7, 7])

    def reset_parameters(self):
        nn.init.normal_(self.graph_conv.weight, std=0.01)
        nn.init.constant_(self.graph_conv.bias, 0)

    def forward(self, input, adj): #input: B*2048*7*7,      adj: B*B
        batch, channel, height, width = input.size(0), input.size(1), input.size(2), input.size(3)
        input_norm = self.norm(input)
        tmp = self.graph_conv(input_norm)
        output = torch.mm(adj, tmp.view(batch, -1)).view(batch, self.out_features, height, width)+input
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.feat_dim = 2048
        self.gcn_layer = GraphConvolution(self.feat_dim, self.feat_dim)

    def forward(self, input_features, adj_mat):
        batch, channel, height, width = input_features.size(0), input_features.size(1), input_features.size(2), input_features.size(3)
        input_features_reshape = input_features.view(batch, -1).contiguous()

        # cosine similarity
        dot_product_mat = torch.mm(input_features_reshape, torch.transpose(input_features_reshape, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(input_features_reshape * input_features_reshape, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        cos_sim_mat = dot_product_mat / len_mat

        adj_mat = adj_mat.to(cos_sim_mat.device)
        new_adj_mat = adj_mat * cos_sim_mat

        gcn_ft = self.gcn_layer(input_features, new_adj_mat)
        return gcn_ft
