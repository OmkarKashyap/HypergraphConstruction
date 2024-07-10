import torch
import torch.nn as nn
import torch.optim as optim

# class Optim:
#     def __init__():
#         pass
    
#     def Adam(self):
#         pass

optimizers = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'lbfgs': torch.optim.LBFGS,
    'adamw' : torch.optim.AdamW,
    'rmsprop' : torch.optim.RMSprop
}