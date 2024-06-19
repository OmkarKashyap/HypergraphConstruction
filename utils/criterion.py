import torch
import torch.nn as nn

# class Criterion:
#     def __init__(self):
#         pass
    
#     def BCELoss(self):
#         pass

criterion = {
    'BCELoss': nn.BCELoss(),
    'MSELoss': nn.MSELoss(),
    'CrossEntropyLoss': nn.CrossEntropyLoss(),
    'NLLLoss': nn.NLLLoss(),
    'KLDivLoss': nn.KLDivLoss(),
    'L1Loss': nn.L1Loss(),    
}