import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math

class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, value_in_features, value_out_features, costate_in_features, costate_out_features,
                 branch_in_features, branch_out_features, num_hidden_layers, value_hidden_features,
                 costate_hidden_features, branch_hidden_features, outermost_linear=False,
                 nonlinearity='relu', weight_init=None):
        super().__init__()

        params = torch.tensor(0.1, requires_grad=True)
        self.value_params = nn.Parameter(params)
        self.costate_params = nn.Parameter(params)
        self.branch_params = nn.Parameter(params)

        self.first_layer_init = None

        self.num_hidden_layers = num_hidden_layers

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None),
                         'gelu': (nn.GELU(), init_weights_normal, None)}

        self.nl, self.nl_weight_init, self.first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = self.nl_weight_init

        # =====================adaptive activation tuning===============================
        # value net setting
        self.value_net = []
        self.value_net.append(nn.Sequential(BatchLinear(value_in_features, value_hidden_features)))

        for i in range(self.num_hidden_layers):
            self.value_net.append(nn.Sequential(BatchLinear(value_hidden_features, value_hidden_features)))

        if outermost_linear:
            self.value_net.append(nn.Sequential(BatchLinear(value_hidden_features, value_out_features)))
        else:
            self.value_net.append(nn.Sequential(BatchLinear(value_hidden_features, value_out_features)))

        self.value_net = nn.Sequential(*self.value_net)
        if self.weight_init is not None:
            self.value_net.apply(self.weight_init)

        if self.first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.value_net[0].apply(self.first_layer_init)

        # costate net setting
        self.costate_net = []
        self.costate_net.append(nn.Sequential(BatchLinear(costate_in_features, costate_hidden_features)))

        for i in range(self.num_hidden_layers):
            self.costate_net.append(nn.Sequential(BatchLinear(costate_hidden_features, costate_hidden_features)))

        if outermost_linear:
            self.costate_net.append(nn.Sequential(BatchLinear(costate_hidden_features, costate_out_features)))
        else:
            self.costate_net.append(nn.Sequential(BatchLinear(costate_hidden_features, costate_out_features)))

        self.costate_net = nn.Sequential(*self.costate_net)
        if self.weight_init is not None:
            self.costate_net.apply(self.weight_init)

        if self.first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.costate_net[0].apply(self.first_layer_init)

        # branch net setting
        self.branch_net = []
        self.branch_net.append(nn.Sequential(BatchLinear(branch_in_features, branch_hidden_features)))

        for i in range(self.num_hidden_layers):
            self.branch_net.append(nn.Sequential(BatchLinear(branch_hidden_features, branch_hidden_features)))

        if outermost_linear:
            self.branch_net.append(nn.Sequential(BatchLinear(branch_hidden_features, branch_out_features)))
        else:
            self.branch_net.append(nn.Sequential(BatchLinear(branch_hidden_features, branch_out_features)))

        self.branch_net = nn.Sequential(*self.branch_net)
        if self.weight_init is not None:
            self.branch_net.apply(self.weight_init)

        if self.first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.branch_net[0].apply(self.first_layer_init)

    def forward(self, coords_vn, coords_cn, input_fun, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        input_vn = coords_vn
        input_cn = coords_cn
        input_branch = input_fun

        for i in range(self.num_hidden_layers + 1):
            input_vn = self.nl(10 * self.value_params * self.value_net[i](input_vn))
            input_cn = self.nl(10 * self.costate_params * self.costate_net[i](input_cn))
            input_branch = self.nl(10 * self.branch_params * self.branch_net[i](input_branch))

        output_vn = torch.sum(self.value_net[-1](input_vn) * self.branch_net[-1](input_branch), dim=-1, keepdim=True)
        output_cn = self.costate_net[-1](input_cn)

        return output_vn, output_cn


class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, value_in_features=2, value_out_features=1, costate_in_features=2, costate_out_features=1,
                 branch_in_features=2, branch_out_features=1, type='sine', mode='mlp',
                 value_hidden_features=256, costate_hidden_features=256, branch_hidden_features=256,
                 num_hidden_layers=3, batch_num_features=10000, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(value_in_features=value_in_features, value_out_features=value_out_features,
                           costate_in_features=costate_in_features, costate_out_features=costate_out_features,
                           branch_in_features=branch_in_features, branch_out_features=branch_out_features,
                           num_hidden_layers=num_hidden_layers, value_hidden_features=value_hidden_features,
                           costate_hidden_features=costate_hidden_features, branch_hidden_features=branch_hidden_features,
                           outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_vn = model_input['coords_vn'].clone().detach().requires_grad_(True)
        coords_cn = model_input['coords_cn'].clone().detach().requires_grad_(True)
        input_fun = model_input['input_fun'].clone().detach().requires_grad_(True)

        output_vn, output_cn = self.net(coords_vn, coords_cn, input_fun)
        return {'model_in_vn': coords_vn, 'model_in_cn': coords_cn, 'input_fun': input_fun,
                'model_out_vn': output_vn, 'model_out_cn': output_cn}


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

