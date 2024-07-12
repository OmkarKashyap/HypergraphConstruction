import sklearn
from sklearn.cluster import DBSCAN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.lda_hypergraph import SemanticHypergraphModel
from models.knn_hg import KNNHG

class HGConstruct(nn.Module):
    def __init__(self,args, config) -> None:
        super(HGConstruct, self). __init__()
        self.args=args
        self.clusters = None
        self.knnhg =KNNHG(args, config)
        # self.attention_context_vector = nn.Parameter(torch.empty(self.output_size))

    def forward(self, inputs):
        knn_hg = self.knnhg(inputs).to(self.args.device)
        
        return knn_hg

class VertexConv(nn.Module):
    def __init__(self, args, in_feats, out_feats):
        super(VertexConv, self).__init__()
        self.args=args
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.attention = nn.Linear(in_feats, out_feats, bias=False)
        self.projection = nn.Linear(in_feats, out_feats, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

    def forward(self, feature_mat, inc_mat, prev_edge_features=None):
        # feature_mat: (m x d) -> m: number of nodes, d: input feature dimension
        # inc_mat: (m x e) -> m: number of nodes, e: number of edges

        b, m, d = feature_mat.size()
        b, _, e = inc_mat.size()

        self.attention = self.attention.to(self.args.device)
        self.projection=self.projection.to(self.args.device)
        # Compute attention scores for each node for each edge
        node_attention_scores = self.attention(feature_mat) # (m x out_feats)

        # Compute attention scores for each edge by aggregating node scores
        edge_attention_scores = torch.bmm(inc_mat.transpose(1, 2), node_attention_scores) # (e x out_feats)

        # Apply softmax to get attention weights for each edge
        attention_weights = F.softmax(edge_attention_scores, dim=1) # (e x out_feats)

        # Compute weighted node features
        weighted_node_features = torch.einsum('bme,bmd->bmed', inc_mat, feature_mat)# (m x e x d)

        # Aggregate weighted node features to get edge features
        edge_features = torch.einsum('bmed,bed->bed', weighted_node_features, attention_weights)  # (e x d)

        # Project aggregated edge features to the output dimension
        edge_features = self.projection(edge_features) # (e x out_feats)

        # If previous edge features are provided, combine them with the new ones based on alpha
        if prev_edge_features is not None:
            edge_features = self.alpha * prev_edge_features + (1 - self.alpha) * edge_features

        return edge_features

class EdgeConv(nn.Module):
    def __init__(self, args, in_feats, out_feats):
        super(EdgeConv, self).__init__()
        self.args=args
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.attention = nn.Linear(in_feats, out_feats, bias=False)
        self.projection = nn.Linear(in_feats, out_feats, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

    def forward(self, inc_mat, edge_features, prev_node_features=None):
        # inc_mat: (m x e) -> m: number of nodes, e: number of edges
        # edge_features: (e x d) -> e: number of edges, d: edge feature dimension

        b, m, e = inc_mat.size()

        self.attention = self.attention.to(self.args.device)
        self.projection=self.projection.to(self.args.device)

        # Compute attention scores for each edge
        edge_attention_scores = self.attention(edge_features)  # (e x out_feats)

        # Aggregate edge attention scores to get node attention scores
        # node_attention_scores = inc_mat @ edge_attention_scores  
        node_attention_scores = torch.matmul(inc_mat, edge_attention_scores) # (m x out_feats)

        # Apply softmax to get attention weights for each node
        attention_weights = F.softmax(node_attention_scores, dim=1)  # (m x out_feats)

        # Weighted aggregation of edge features to get new node features
        weighted_edge_features = torch.einsum('bed,bme->bmd', edge_features, inc_mat)  # (m x d)

        # Project aggregated node features to the output dimension
        node_features = self.projection(weighted_edge_features)  # (m x out_feats)

        # If previous node features are provided, combine them with the new ones based on alpha
        if prev_node_features is not None:
            node_features = self.alpha * prev_node_features + (1 - self.alpha) * node_features

        return node_features
        

class HypergraphEdgeAggregation(nn.Module):
    def __init__(self, args, in_features, out_features):
        super(HypergraphEdgeAggregation, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(in_features, 1, bias=False)

    def forward(self, edge_features):
        """
        Perform edge aggregation for hypergraph representation using attention mechanism.
        
        :param edge_features: Aggregated node features for each edge (e x d)
        :return: Aggregated edge features (out_features)
        """

        self.projection=self.projection.to(self.args.device)
        self.attention =self.attention.to(self.args.device)
        # Compute attention scores for each edge
        attention_scores = self.attention(edge_features)  # (b x e x 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (b x e x 1)

        # Compute weighted sum of edge features
        weighted_edge_features = edge_features * attention_weights  # (b x e x d)
        pooled_features = torch.sum(weighted_edge_features, dim=1)  # (b x d,)

        # Project to output features
        output_features = self.projection(pooled_features)  # (b x out_features,)
        
        return output_features


class MessagePassing(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.dim_in = args.dim_in
        self.num_layers = config.n_layers
        self.vc = VertexConv(args, self.dim_in, self.dim_in)
        self.construct = HGConstruct(args, config)
        self.ec = EdgeConv(args, self.dim_in, self.dim_in)
    
    def forward(self, inputs):
        # features = features.squeeze(0)
        features = inputs[0]
        edge_feats = self.vc(features)
        node_feats = self.ec(edge_feats, features)
        for _ in range(1, self.num_layers):
            
            edge_feats = self.vc(features, edge_feats)
            node_feats = self.ec(edge_feats, node_feats)
        
        return node_feats, edge_feats

class HGConv(nn.Module):
    def  __init__(self, args, config) -> None:
        super(). __init__()
        self.args=args
        self.dim_in = args.dim_in
        self.n_categories = args.n_categories  # Number of output categories
        self.has_bias = True  # Whether to use bias in the fc layer
        self.dropout = nn.Dropout(config.dropout_rate)
        self.vc = VertexConv(args, self.dim_in, self.dim_in)
        self.ec = HypergraphEdgeAggregation(args, self.dim_in, self.dim_in)
        self.fc = nn.Linear(self.dim_in, self.n_categories, bias=self.has_bias)
        self.activation = nn.LogSoftmax(dim=-1)  # Activation function for the output layer
    
    def forward(self, node_feats, edge_feats):
        self.fc=self.fc.to(self.args.device)
        edge_feat = self.vc(node_feats, edge_feats)
        logits = self.ec(edge_feat)
        logits = self.fc((logits))
        return logits
    
class HGScanLayer(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.msg_pass = MessagePassing(args, config)
        self.conv = HGConv(args, config)
    
    def forward(self, inputs):
        node_feats, edge_feats = self.msg_pass(inputs)
        logits = self.conv(node_feats, edge_feats)
        return logits


        

