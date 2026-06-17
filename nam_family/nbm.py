from itertools import combinations
import torch
import torch.nn as nn
from .utils import ChannelWiseLinear, GAMOutputLayer, get_selector_layer


def get_mlps(nn_in_dim, nn_out_dim, hidden_dims=(256, 128, 128), dropout_rate=0., batch_norm=True):
    # Hidden layer
    layers = []
    in_d = nn_in_dim
    for n in range(len(hidden_dims)):
        layers += [nn.Linear(in_features=in_d, out_features=hidden_dims[n])]
        if batch_norm:
            layers += [nn.BatchNorm1d(num_features=hidden_dims[n])]
        if dropout_rate > 0.:
            layers += [nn.Dropout(p=dropout_rate)]
        layers += [nn.ReLU()]
        in_d = hidden_dims[n]

    # Last layer
    layers += [nn.Linear(in_features=hidden_dims[-1], out_features=nn_out_dim)]
    if batch_norm:
        layers += [nn.BatchNorm1d(nn_out_dim)]
    layers += [nn.ReLU()]
    return nn.Sequential(*layers)


class NBMLayer(nn.Module):
    def __init__(self, one_input_num, two_input_num, bases_num, hidden_dims=(256, 128, 128), dropout_rate=0.,
                 bases_dropout=0., batch_norm=True):
        super(NBMLayer, self).__init__()

        self.one_input_num, self.two_input_num = one_input_num, two_input_num

        self.nbm1 = get_mlps(1, bases_num, hidden_dims, dropout_rate, batch_norm) if one_input_num > 0 else None
        self.nbm2 = get_mlps(2, bases_num, hidden_dims, dropout_rate, batch_norm) if two_input_num > 0 else None

        self.b_dropout = nn.Dropout(bases_dropout) if bases_dropout > 0. else nn.Identity()

        # Linear projection of MLP outputs for each shape function
        self.shape_linear = ChannelWiseLinear(bases_num, 1, one_input_num + two_input_num, bias=False)

    # x1: shape=(minibatch, one_input_num), x2: shape=(minibatch, two_input_num, 2),
    # return: shape=(minibatch, shape_num)
    def forward(self, x1, x2):
        y = []
        # One-input NN
        if self.one_input_num > 0:
            x1 = x1.unsqueeze(dim=2)  # shape = (minibatch, one_input_num, 1)
            y1 = self.nbm1(x1.reshape(-1, x1.shape[-1])).reshape(x1.shape[0], x1.shape[1], -1)
            y1 = self.b_dropout(y1)  # shape = (minibatch, one_input_num, bases_num)
            y.append(y1)

        # Two-input NN
        if self.two_input_num > 0:
            y2 = self.nbm2(x2.reshape(-1, x2.shape[-1])).reshape(x2.shape[0], x2.shape[1], -1)
            y2 = self.b_dropout(y2)  # shape = (minibatch, two_input_num, bases_num)
            y.append(y2)

        y = torch.cat(y, dim=1)
        y = self.shape_linear(self.b_dropout(y))  # shape = (minibatch, shape_num, 1)

        return y.squeeze(dim=2)


class NBM(nn.Module):
    def __init__(self, in_dim, out_dim, bases_num, hidden_dims=(256, 128, 128), dropout_rate=0., bases_dropout=0.,
                 batch_norm=True, pairwise=False):
        super(NBM, self).__init__()

        self.feature_outputs = None
        self.shape_outputs = None

        self.in_dim, self.out_dim = in_dim, out_dim
        self.bases_num = bases_num
        self.one_input_shape_num = in_dim
        if pairwise:
            self.two_input_shape_num = len(list(combinations(range(self.in_dim), 2)))
            self.pairwise_indices = torch.tensor(list(combinations(range(self.in_dim), 2)), dtype=torch.int64).flatten()
        else:
            self.pairwise_indices = None
            self.two_input_shape_num = 0
        self.shape_num = self.one_input_shape_num + self.two_input_shape_num

        self.nbm = NBMLayer(self.one_input_shape_num, self.two_input_shape_num, bases_num, hidden_dims,
                            dropout_rate, bases_dropout, batch_norm)

        self.gam_out_layer = GAMOutputLayer(self.shape_num, self.out_dim)

    # x: shape=(minibatch, in_dim), return: shape=(minibatch, out_dim)
    def forward(self, x):
        # One-input
        x1 = x.unsqueeze(dim=2)  # shape = (minibatch, in_dim, 1)
        # Two-input
        x2 = None
        if self.two_input_shape_num > 0:
            # shape = (minibatch, pairwise_num, 2)
            x2 = x[:, self.pairwise_indices].reshape(-1, self.two_input_shape_num, 2)

        self.feature_outputs = self.nbm(x1, x2)
        self.shape_outputs = self.feature_outputs
        return self.gam_out_layer(self.shape_outputs)

    def get_input_feature_index(self):
        uni_indices = torch.tensor([i for i in range(self.in_dim)])
        if self.pairwise_indices is not None:
            pair_indices = self.pairwise_indices.reshape(-1, 2)
        else:
            pair_indices = torch.tensor([], dtype=torch.int64)
        return uni_indices, pair_indices


class NBMFS(nn.Module):
    def __init__(self, in_dim, out_dim, bases_num, one_input_shape_num, two_input_shape_num, feature_sel='Entmax',
                 hidden_dims=(256, 128, 128), dropout_rate=0., bases_dropout=0., batch_norm=True):
        super(NBMFS, self).__init__()

        self.feature_outputs = None
        self.shape_outputs = None

        self.in_dim, self.out_dim = in_dim, out_dim
        self.bases_num = bases_num
        self.one_input_shape_num = one_input_shape_num
        self.two_input_shape_num = two_input_shape_num
        self.shape_num = self.one_input_shape_num + self.two_input_shape_num

        sel_num = self.one_input_shape_num + 2 * self.two_input_shape_num
        self.sel_layer = get_selector_layer(feature_sel, sel_num, in_dim)

        self.nbm = NBMLayer(self.one_input_shape_num, self.two_input_shape_num, bases_num, hidden_dims,
                            dropout_rate, bases_dropout, batch_norm)

        self.gam_out_layer = GAMOutputLayer(self.shape_num, self.out_dim)

    # x: shape=(minibatch, in_dim), return: shape=(minibatch, out_dim)
    def forward(self, x, temperature=None):
        # Feature selector
        if temperature is not None:
            z = self.sel_layer(x, temperature)  # shape=(minibatch, sel_num)
        else:
            z = x[:, self.sel_layer.get_index()]

        x1 = None
        if self.one_input_shape_num > 0:
            x1 = z[:, :self.one_input_shape_num]
        x2 = None
        if self.two_input_shape_num:
            x2 = z[:, self.one_input_shape_num:].reshape(-1, self.two_input_shape_num, 2)

        self.feature_outputs = self.nbm(x1, x2)
        self.shape_outputs = self.feature_outputs
        return self.gam_out_layer(self.shape_outputs)

    def get_input_feature_index(self):
        uni_indices = self.sel_layer.get_index()[:self.one_input_shape_num]
        pair_indices = self.sel_layer.get_index()[self.one_input_shape_num:]
        return uni_indices, pair_indices.reshape(-1, 2)
