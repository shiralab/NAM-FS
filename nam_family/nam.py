from itertools import combinations
import torch
import torch.nn as nn
from .utils import ChannelWiseLinear, ExU, GAMOutputLayer
from .utils import get_selector_layer


class NAMBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(NAMBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        y = x.reshape(x.shape[0], -1)
        y = self.bn(y)
        return y.reshape(x.shape[0], x.shape[1], -1)


def get_nam_hidden_layer_list(nn_in_dim, nn_out_dim, nn_num, use_exu=False, dropout_rate=0., batch_norm=True):
    if use_exu:
        layers = [ExU(nn_in_dim, nn_out_dim, nn_num)]
        if batch_norm:
            layers += [NAMBatchNorm1d(nn_out_dim * nn_num)]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
    else:
        layers = [ChannelWiseLinear(nn_in_dim, nn_out_dim, nn_num)]
        if batch_norm:
            layers += [NAMBatchNorm1d(nn_out_dim * nn_num)]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers += [nn.ReLU()]

    return layers


class NAMLayer(nn.Module):
    def __init__(self, one_input_nn_num, two_input_nn_num, hidden_dims=(64, 64, 32), use_exu=False, dropout_rate=0.,
                 batch_norm=True):
        super(NAMLayer, self).__init__()

        nn_num = one_input_nn_num + two_input_nn_num

        # First hidden layer
        # One-input NN
        if one_input_nn_num > 0:
            layers = get_nam_hidden_layer_list(1, hidden_dims[0], one_input_nn_num, use_exu, dropout_rate, batch_norm)
            self.h1 = nn.Sequential(*layers)
        else:
            self.h1 = None
        # Two-input NN
        if two_input_nn_num > 0:
            layers = get_nam_hidden_layer_list(2, hidden_dims[0], two_input_nn_num, use_exu, dropout_rate, batch_norm)
            self.h2 = nn.Sequential(*layers)
        else:
            self.h2 = None

        # Second to final hidden layers
        layers = []
        for n in range(len(hidden_dims) - 1):
            layers += get_nam_hidden_layer_list(hidden_dims[n], hidden_dims[n+1], nn_num, use_exu, dropout_rate,
                                                batch_norm)
        # Last linear layer
        layers += [ChannelWiseLinear(hidden_dims[-1], 1, nn_num)]
        self.h = nn.Sequential(*layers)

    # x1: shape=(minibatch, one_input_nn_num), x2: shape=(minibatch, two_input_nn_num, 2),
    # return: shape=(minibatch, nn_num)
    def forward(self, x1, x2):
        y = []
        # One input NN: first hidden layer
        if self.h1 is not None:
            y += [self.h1(x1.unsqueeze(dim=2))]  # shape = (minibatch, one_input_nn_num, hidden_dims[0])

        # Two input NN: first hidden layer
        if self.h2 is not None:
            y += [self.h2(x2)]  # shape = (minibatch, two_input_nn_num, hidden_dims[0])

        # Concat
        y = torch.cat(y, dim=1)  # shape = (minibatch, nn_num, hidden_dims[0])
        # Second hidden layers to output layer
        y = self.h(y)  # shape = (minibatch, nn_num, 1)
        return y.squeeze(dim=2)


class NAM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64, 32), use_exu=False, dropout_rate=0., feature_dropout=0.,
                 batch_norm=True, pairwise=False):
        super(NAM, self).__init__()

        self.feature_outputs = None
        self.shape_outputs = None

        self.in_dim, self.out_dim = in_dim, out_dim
        self.one_input_shape_num = in_dim
        if pairwise:
            self.two_input_shape_num = len(list(combinations(range(self.in_dim), 2)))
            self.pairwise_indices = torch.tensor(list(combinations(range(self.in_dim), 2)), dtype=torch.int64).flatten()

        else:
            self.pairwise_indices = None
            self.two_input_shape_num = 0
        self.shape_num = self.one_input_shape_num + self.two_input_shape_num

        self.nam_layer = NAMLayer(self.one_input_shape_num, self.two_input_shape_num, hidden_dims, use_exu,
                                  dropout_rate, batch_norm)

        self.f_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0. else nn.Identity()

        self.gam_out_layer = GAMOutputLayer(self.shape_num, self.out_dim)

    # x: shape=(minibatch, in_dim), return: shape=(minibatch, out_dim)
    def forward(self, x):
        if self.two_input_shape_num > 0:
            x2 = x[:, self.pairwise_indices].reshape(-1, self.two_input_shape_num, 2)
        else:
            x2 = None

        self.feature_outputs = self.nam_layer(x, x2)  # shape=(minibatch, shape_num)
        self.shape_outputs = self.f_dropout(self.feature_outputs)
        return self.gam_out_layer(self.shape_outputs)

    def get_input_feature_index(self):
        uni_indices = torch.tensor([i for i in range(self.in_dim)])
        if self.pairwise_indices is not None:
            pair_indices = self.pairwise_indices.reshape(-1, 2)
        else:
            pair_indices = torch.tensor([], dtype=torch.int64)
        return uni_indices, pair_indices


class NAMFS(nn.Module):
    def __init__(self, in_dim, out_dim, one_input_shape_num, two_input_shape_num, feature_sel='Entmax',
                 hidden_dims=(64, 64, 32), use_exu=False, dropout_rate=0., feature_dropout=0., batch_norm=True):
        super(NAMFS, self).__init__()

        self.feature_outputs = None
        self.shape_outputs = None

        self.in_dim, self.out_dim = in_dim, out_dim
        self.one_input_shape_num = one_input_shape_num
        self.two_input_shape_num = two_input_shape_num
        self.shape_num = self.one_input_shape_num + self.two_input_shape_num

        sel_num = self.one_input_shape_num + 2 * self.two_input_shape_num
        self.sel_layer = get_selector_layer(feature_sel, sel_num, in_dim)

        self.nam_layer = NAMLayer(self.one_input_shape_num, self.two_input_shape_num, hidden_dims, use_exu,
                                  dropout_rate, batch_norm)

        self.f_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0. else nn.Identity()

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

        # nam_family Layer
        self.feature_outputs = self.nam_layer(x1, x2)  # shape=(minibatch, shape_num)
        self.shape_outputs = self.f_dropout(self.feature_outputs)
        return self.gam_out_layer(self.shape_outputs)

    def get_input_feature_index(self):
        uni_indices = self.sel_layer.get_index()[:self.one_input_shape_num]
        pair_indices = self.sel_layer.get_index()[self.one_input_shape_num:]
        return uni_indices, pair_indices.reshape(-1, 2)
