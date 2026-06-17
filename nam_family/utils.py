import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import entmax15


class Entmax15SelectorLayer(nn.Module):
    def __init__(self, sel_num, in_dim):
        super(Entmax15SelectorLayer, self).__init__()
        # parameters of feature selection logit
        self.sel_logit = nn.init.uniform_(nn.Parameter(torch.zeros((sel_num, in_dim))))

    def get_index(self):
        return torch.argmax(self.sel_logit, dim=1)

    # x: shape=(minibatch, in_dim)
    # return: weighted values, shape=(minibatch, sel_num)
    def forward(self, x, temperature):
        m = entmax15(self.sel_logit / temperature, dim=1)
        return F.linear(x, m)


class GumbelSelectorLayer(nn.Module):
    def __init__(self, sel_num, in_dim):
        super(GumbelSelectorLayer, self).__init__()
        # parameters of feature selection logit
        self.sel_logit = nn.init.uniform_(nn.Parameter(torch.zeros((sel_num, in_dim))))

    def get_index(self):
        return torch.argmax(self.sel_logit, dim=1)

    # x: shape=(minibatch, in_dim)
    # return: weighted values, shape=(minibatch, sel_num)
    def forward(self, x, temperature):
        m = F.gumbel_softmax(self.sel_logit, tau=temperature, dim=1)
        return F.linear(x, m)


class RandomSelectorLayer(nn.Module):
    def __init__(self, sel_num, in_dim):
        super(RandomSelectorLayer, self).__init__()
        self.fea_idx = torch.randint(low=0, high=in_dim, size=(sel_num,))

    def get_index(self):
        return self.fea_idx

    # x: shape=(minibatch, in_dim)
    # return: randomly selected feature values, shape=(minibatch, sel_num)
    def forward(self, x, temperature):
        return x[:, self.fea_idx]


class ExU(nn.Module):
    def __init__(self, nn_in_dim, nn_out_dim, nn_num, init_weight_mean=4.):
        super(ExU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(nn_num, nn_in_dim, nn_out_dim))
        self.bias = nn.Parameter(torch.Tensor(nn_num, 1, nn_in_dim))
        self.init_weight_mean = init_weight_mean
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=self.init_weight_mean, std=0.5)
        nn.init.normal_(self.bias, mean=0, std=0.5)

    # x: shape=(minibatch, nn_num, nn_in_dim)
    # return: shape=(minibatch, nn_num, nn_out_dim)
    def forward(self, x):
        x = torch.permute(x, (1, 0, 2))  # shape = (nn_num, minibatch, in_dim)
        y = torch.bmm(x - self.bias, torch.exp(self.weight))
        # ReLU-n (n=1)
        n = 1
        y = torch.clamp(y, 0, n)
        return torch.permute(y, (1, 0, 2))


class ChannelWiseLinear(nn.Module):
    def __init__(self, nn_in_dim, nn_out_dim, channel_num, bias=True):
        super(ChannelWiseLinear, self).__init__()
        self.nn_in_dim = nn_in_dim
        self.weight = nn.Parameter(torch.Tensor(channel_num, nn_in_dim, nn_out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(channel_num, 1, nn_out_dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.nn_in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    # x: shape=(minibatch, channel_num, nn_in_dim)
    # return: shape=(minibatch, channel_num, nn_out_dim)
    def forward(self, x):
        x = torch.permute(x, (1, 0, 2))  # shape = (channel_num, minibatch, in_dim)
        y = torch.bmm(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return torch.permute(y, (1, 0, 2))


class GAMOutputLayer(nn.Module):
    def __init__(self, shape_func_num, out_dim):
        super(GAMOutputLayer, self).__init__()

        self.in_dim = shape_func_num
        if out_dim == 1:
            self.weight = None
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_dim, shape_func_num))
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_dim)
        if self.weight is not None:
            nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    # x: shape = (minibatch, shape_func_num)
    def forward(self, x):
        if self.weight is None:
            return x.sum(dim=1, keepdim=True) + self.bias
        else:
            return F.linear(x, self.weight, self.bias)


def get_selector_layer(type_name, sel_num, in_dim):
    if type_name == 'Entmax':
        sel_layer = Entmax15SelectorLayer(sel_num, in_dim)
    elif type_name == 'Gumbel':
        sel_layer = GumbelSelectorLayer(sel_num, in_dim)
    elif type_name == 'Random':
        sel_layer = RandomSelectorLayer(sel_num, in_dim)
    else:
        sel_layer = None
        print('Error: undefined selector layer name...')
        sys.exit(1)
    return sel_layer


# return: shape=(shape_num, out_dim)
# def compute_feature_contribution(shape_outputs, weight=None):
#     if weight is None:
#         return torch.sum(torch.abs(shape_outputs), dim=0).unsqueeze(dim=1)
#     else:
#         shape_abs_sum = torch.sum(torch.abs(shape_outputs), dim=0)
#         contrib = shape_abs_sum * torch.abs(weight)
#         return contrib.T
