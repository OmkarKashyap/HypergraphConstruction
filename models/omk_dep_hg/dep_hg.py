import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from models.omk_dep_hg.dep_hg_utils import DependencyGraph, GNNLayer, CommunityDetection, GCN

class DependencyHG(nn.Module):
    def __init__(self, args, config):
        super(DependencyHG, self).__init__()
        
        self.args =args
        self.graph_model = DependencyGraph(args)
        self.gcn = GCN(args, config)
        self.community = CommunityDetection(args)
        
    def forward(self, inputs):
        feats, tokens, aspect, pos, post, head, deprel, sen_len, adk, pos_mask, word_mask, aspect_pos_start, aspect_pos_end,plain_text, text_list= inputs
        
        # Move tensors to the same device
        feats = feats.to(self.args.device)
        tokens = [t.to(self.args.device) for t in tokens]
        word_mask = [wm.to(self.args.device) for wm in word_mask]

        # Process through graph model and GNN layer
        adj = self.graph_model(inputs)
        adj = adj.to(self.args.device)  # Ensure adj is on the same device
        out = self.gcn(feats, adj, word_mask)
        out = out.to(self.args.device)
        # Community detection
        communities, incidence_matrix = self.community.louvain(adj, out, tokens, text_list, word_mask)
        
        return incidence_matrix
