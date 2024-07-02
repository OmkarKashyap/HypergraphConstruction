import sklearn
from sklearn.cluster import DBSCAN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.omk_dep_hg.dep_hg import DependencyHG
from models.lda_hypergraph import SemanticHypergraphModel

class HGConstruct(nn.Module):
    def __init__(self, eps, min_samples, args) -> None:
        super(HGConstruct, self). __init__()
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = None
        self.dep = DependencyHG(args)
        self.lda = SemanticHypergraphModel(args)
        # self.attention_context_vector = nn.Parameter(torch.empty(self.output_size))


    def make_incidence_matrix(self, hyperedges, n, nc):
        '''
        returns incidence matrix from the clusters in DBSCAN
        :param hyperedges: List[nodes in each hyperedge]
        :param n: number of nodes
        :param nc: number of clusters
        '''
        incidence_matrix = torch.zeros(n, nc, dtype=torch.float)

        # Populate the incidence matrix
        for cluster_idx, cluster in enumerate(hyperedges):
            for node_idx in cluster:
                incidence_matrix[node_idx, cluster_idx] = 1.0
        
        return incidence_matrix

    def cluster(self, features):
        '''
        perform DBSCAN over the nodes, one dimension at a time.
        return an incidence matrix with each cluster as a hyperedge.
        :param ids: indices selected during train/valid/test, torch.LongTensor.
        :param features: assumed it is a tensor. (N, k, d)
        '''
        print(features.shape)
        num_nodes = features.size()[0]
        dim_len = features.size()[1]
        num_clusters = 0
        hyperedges = []

        for dim in range(dim_len):
            # print(features.squeeze(0)[:,dim].reshape(-1, 1).shape)\
            # reshaped_tensor = features.view(85, 1)
            # print(reshaped_tensor.shape)
            current_dim_emb = features.squeeze(0)[:,dim].reshape(-1, 1)
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = db.fit_predict(current_dim_emb)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            clusters_now = len(unique_labels)
            dim_clusters = [np.where(labels == label)[0].tolist() for label in unique_labels]
            num_clusters += clusters_now
            if dim_clusters:
                hyperedges.extend(dim_clusters)
    
        incidence_matrix = self.make_incidence_matrix(hyperedges, num_nodes, num_clusters)
    # batch_incidence_matrix = torch.stack(batch_incidence_matrix)
        return incidence_matrix

    def forward(self, inputs, inc_mat):
        lda_hg = self.lda(inputs)
        dep_hg = self.dep(inputs)
        final_inc = torch.cat((dep_hg, inc_mat, lda_hg), dim=2)
        return final_inc

        
# class VertexConv(nn.Module):
#     def  __init__(self, in_feats, out_feats):
#         super(). __init__()
#         self.in_dim = in_feats
#         self.out_dim = out_feats
#         self.attention = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
#         self.projection = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
    
#     def forward(self, feature_mat, inc_mat):
#         m, d = feature_mat.size()
#         _, e = inc_mat.size()

#         attention_scores = self.attention(feature_mat)  # (m x out_features)
        
#         # Aggregate node features to get edge features
#         X_agg = inc_mat.T @ feature_mat  # (e x d)
        
#         return X_agg

class VertexConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(VertexConv, self).__init__()
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

        # Compute attention scores for each node for each edge
        node_attention_scores = self.attention(feature_mat)  # (m x out_feats)

        # Compute attention scores for each edge by aggregating node scores
        edge_attention_scores = torch.bmm(inc_mat.transpose(1, 2), node_attention_scores)  # (e x out_feats)

        # Apply softmax to get attention weights for each edge
        attention_weights = F.softmax(edge_attention_scores, dim=1)  # (e x out_feats)

        # Compute weighted node features
        weighted_node_features = torch.einsum('bme,bmd->bmed', inc_mat, feature_mat)  # (m x e x d)

        # Aggregate weighted node features to get edge features
        edge_features = torch.einsum('bmed,bed->bed', weighted_node_features, attention_weights)  # (e x d)

        # Project aggregated edge features to the output dimension
        edge_features = self.projection(edge_features)  # (e x out_feats)

        # If previous edge features are provided, combine them with the new ones based on alpha
        if prev_edge_features is not None:
            edge_features = self.alpha * prev_edge_features + (1 - self.alpha) * edge_features

        return edge_features

class EdgeConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(EdgeConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.attention = nn.Linear(in_feats, out_feats, bias=False)
        self.projection = nn.Linear(in_feats, out_feats, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

    def forward(self, inc_mat, edge_features, prev_node_features=None):
        # inc_mat: (m x e) -> m: number of nodes, e: number of edges
        # edge_features: (e x d) -> e: number of edges, d: edge feature dimension

        b, m, e = inc_mat.size()

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
        

# class HypergraphEdgeAggregation(nn.Module):
#     def  __init__(self, in_features, out_features):
#         super(HypergraphEdgeAggregation, self). __init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.projection = nn.Linear(in_features, out_features)
    
#     def forward(self, edge_features):
#         """
#         Perform edge aggregation for hypergraph representation.
        
#         :param edge_features: Aggregated node features for each edge (e x d)
#         :return: Aggregated edge features (out_features)
#         """
#         # Perform mean pooling across all edges
#         pooled_features = torch.mean(edge_features, dim=0)  # (d,) -> (out_features,)
        
#         # Project to output features
#         output_features = self.projection(pooled_features)  # (out_features,)
        
#         return output_features

class HypergraphEdgeAggregation(nn.Module):
    def __init__(self, in_features, out_features):
        super(HypergraphEdgeAggregation, self).__init__()
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
    def __init__(self, args) -> None:
        super().__init__()
        self.eps = args.eps
        self.min_samples = args.min_samples
        self.output_size = args.output_size
        self.dim_in = args.dim_in
        self.hidden_num = args.hidden_num
        self.num_layers = args.n_layers
        self.vc = VertexConv(self.dim_in, self.dim_in)
        self.construct = HGConstruct(args=args, eps=self.eps, min_samples = self.min_samples)
        self.ec = EdgeConv(self.dim_in, self.dim_in)
    
    def forward(self, inputs, inc_mat):
        # features = features.squeeze(0)
        inc_mat = self.construct(inputs, inc_mat)
        features = inputs[0]
        edge_feats = self.vc(features, inc_mat)
        node_feats = self.ec(inc_mat, edge_feats, features)
        for _ in range(1, self.num_layers):
            
            edge_feats = self.vc(features, inc_mat, edge_feats)
            node_feats = self.ec(inc_mat, edge_feats, node_feats)
        
        return node_feats, edge_feats, inc_mat

class HGConv(nn.Module):
    def  __init__(self, args) -> None:
        super(). __init__()

        self.dim_in = args.dim_in
        self.n_categories = args.n_categories  # Number of output categories
        self.has_bias = True  # Whether to use bias in the fc layer
        self.dropout = nn.Dropout(args.dropout_rate)
        self.vc = VertexConv(self.dim_in, self.dim_in)
        self.ec = HypergraphEdgeAggregation(self.dim_in, self.dim_in)
        self.fc = nn.Linear(self.dim_in, self.n_categories, bias=self.has_bias)
        self.activation = nn.LogSoftmax(dim=-1)  # Activation function for the output layer
    
    def forward(self, node_feats, edge_feats, inc_mat):
        edge_feat = self.vc(node_feats, inc_mat, edge_feats)
        logits = self.ec(edge_feat)
        logits = self.fc((logits))
        return logits
    
class HGScanLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.msg_pass = MessagePassing(args)
        self.conv = HGConv(args)
    
    def forward(self, inputs, inc_mat):
        node_feats, edge_feats, inc_mat = self.msg_pass(inputs, inc_mat)
        logits = self.conv(node_feats, edge_feats, inc_mat)
        return logits

class SqueezeEmbedding(nn.Module):
    '''
    Squeeze sequence embedding length to the longest one in the batch
    '''
    def  __init__(self, batch_first=True):
        super(SqueezeEmbedding, self). __init__()
        self.batch_first = batch_first
    
    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        '''unpack'''
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
        if self.batch_first:
            out = out[x_unsort_idx]
        else:
            out = out[:, x_unsort_idx]
        return out

        

