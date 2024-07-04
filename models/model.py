# import torch
# from torch import nn
# from torch.nn import Module
# from models.layers import *

# class HGSCAN(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.n_layers = 2
#         self.dropout = nn.Dropout(0.5)
#         self.hgl = HGScanLayer( )
    
#     def forward(self, features):
#         x = features
#         for i_layer in range(self.n_layers):
#             x = self.hgl(x)
#             x = self.dropout(x)  # Apply dropout after each layer
#         return x

import torch
from torch import nn
from torch.nn import Module
from models.layers import *

class HGSCAN(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.n_layers = args.n_layers
        self.dropout = nn.Dropout(config.dropout_rate)
        # self.hgl = [HGScanLayer() for i in range(self.n_layers)]
        self.hgl = HGScanLayer(args, config)
    
    def forward(self, inputs, inc_mat):
        # x = features
        x = self.hgl(inputs, inc_mat)
        x = self.dropout(x)  # Apply dropout after each layer
        return x.unsqueeze(0)