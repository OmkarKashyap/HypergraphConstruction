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
        self.dropout = nn.Dropout(config.dropout_rate)
        self.hypergraph_NN = HGScanLayer(args, config)
    
    def forward(self, inputs, inc_mat, aspect_emb):
        # x = features
        x = self.hypergraph_NN(inputs, inc_mat, aspect_emb)
        print("Logits shape", x.shape)
        x = self.dropout(x)  # Apply dropout after each layer
        return x