import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import scipy.sparse as sp


class GCN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GCN_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, Laplacian, input_feature):
        """
            adjacency: torch.sparse.FloatTensor
            input_feature: torch.Tensor
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(Laplacian, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('             + str(self.in_features) + ' -> '             + str(self.out_features) + ')'




