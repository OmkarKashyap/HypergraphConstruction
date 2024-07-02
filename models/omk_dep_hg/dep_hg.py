import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from models.omk_dep_hg.dep_hg_utils import DependencyGraph, GNNLayer, CommunityDetection

class DependencyHG(nn.Module):
    def __init__(self, args):
        super(DependencyHG, self).__init__()
        
        self.args =args
        self.graph_model = DependencyGraph(args)
        self.gnn = GNNLayer(args)
        self.community = CommunityDetection(args)
        
    def forward(self, inputs):
        feats, tokens, aspect, pos, post, head, deprel, sen_len, adk, pos_mask, word_mask, aspect_pos_start, aspect_pos_end,plain_text, text_list= inputs
        
        adj = self.graph_model(inputs)
        out = self.gnn(feats, adj, word_mask)
        
        communities, incidence_matrix = self.community.louvain(out, tokens, text_list, word_mask)
        # print(incidence_matrix)
        # return hypergraph
        return incidence_matrix
        
