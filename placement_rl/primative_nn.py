
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class FNN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, out_size):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList([])
        sizes = [input_size] + hidden_layer_sizes+[out_size]
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
        self.apply(weights_init_)
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.layers:
            reset(l)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            y  = layer(y)
            if i != len(self.layers) - 1:
                y = F.relu(y)
        return y









