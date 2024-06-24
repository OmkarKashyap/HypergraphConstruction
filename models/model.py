import torch
from torch import nn
from torch.nn import Module
from models.layers import *

class HGSCAN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_layers = kwargs['n_layers']
        self.dropout = nn.Dropout(kwargs.get('dropout_rate', 0.5))
        self.hgl = HGScanLayer(
        eps = kwargs['eps'],
        min_samples = kwargs['min_samples'],
        output_size = kwargs['output_size'],
        dim_in = kwargs['dim_in'],
        hidden_num = kwargs['hidden_num'],
        ft_dim = kwargs['ft_dim'],
        n_categories = kwargs['n_categories'], # Number of output categories
        has_bias = kwargs.get('has_bias', True),  # Whether to use bias in the fc layer
        droupout_rate = kwargs['droupout_rate']
        )
    
    def forward(self, features):
        x = features

        for i_layer in self.n_layers:
            x = self.hgl(x)
            x = self.dropout(x)  # Apply dropout after each layer
        return x