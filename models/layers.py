import sklearn
from sklearn.cluster import DBSCAN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.omk_dep_hg.dep_hg import DependencyHG
from models.lda_hypergraph import SemanticHypergraphModel
from sklearn.metrics import silhouette_score
import random

class HGConstruct(nn.Module):
    def __init__(self,args, config) -> None:
        super(HGConstruct, self). __init__()
        self.args=args
        self.eps = config.eps
        self.min_samples = config.min_samples
        self.clusters = None
        self.dep = DependencyHG(args, config)
        self.lda = SemanticHypergraphModel(args, config)
        self.silhouette_threshold = 0.6
        # self.attention_context_vector = nn.Parameter(torch.empty(self.output_size))


    def make_incidence_matrix(self, hyperedges, n, nc):
        '''
        returns incidence matrix from the clusters in DBSCAN
        :param hyperedges: List[nodes in each hyperedge]
        :param n: number of nodes
        :param nc: number of clusters
        '''
        incidence_matrix = torch.zeros(n, nc, dtype=torch.float)
        print(len(hyperedges))
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
        lda_hg = self.lda(inputs).to(self.args.device)
        dep_hg = self.dep(inputs).to(self.args.device)
        inc_mat = inc_mat.to(self.args.device)
        final_inc = torch.cat((dep_hg, inc_mat, lda_hg), dim=2)
        return final_inc
    # def cluster(self, features):
    #     '''
    #     perform DBSCAN over the nodes, one dimension at a time.
    #     return an incidence matrix with each cluster as a hyperedge.
    #     :param ids: indices selected during train/valid/test, torch.LongTensor.
    #     :param features: assumed it is a tensor. (N, k, d)
    #     '''
    #     print(features.shape)
    #     num_nodes = features.size()[0]
    #     dim_len = features.size()[1]
    #     num_clusters = 0
    #     hyperedges = []

    #     for dim in range(dim_len):
    #         # print(features.squeeze(0)[:,dim].reshape(-1, 1).shape)\
    #         # reshaped_tensor = features.view(85, 1)
    #         # print(reshaped_tensor.shape)
    #         dim_clusters = []
    #         current_dim_emb = features.squeeze(0)[:,dim].reshape(-1, 1)
    #         db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    #         labels = db.fit_predict(current_dim_emb)
    #         unique_labels = set(labels)
    #         if -1 in unique_labels:
    #             unique_labels.remove(-1)
            
    #         if len(unique_labels) > 1:
    #             silhouette_avg = silhouette_score(current_dim_emb, labels)
                
    #             if silhouette_avg >= self.silhouette_threshold:
    #                 dim_clusters = [np.where(labels == label)[0].tolist() for label in unique_labels]
                    
    #             if dim_clusters:
    #                 num_clusters += len(dim_clusters)
    #                 hyperedges.extend(dim_clusters)
                
    #     print("num_clusters", num_clusters)
    #     incidence_matrix = self.make_incidence_matrix(hyperedges, num_nodes, num_clusters)
    # # batch_incidence_matrix = torch.stack(batch_incidence_matrix)
    #     return incidence_matrix

    # def random_dimension_combinations(self, dim_len, num_combinations, subset_size):
    #     combinations = []
    #     for _ in range(num_combinations):
    #         combination = random.sample(range(dim_len), subset_size)
    #         combinations.append(combination)
    #     return combinations
    
    # def cluster(self, features):
    #     # print(features.shape)
    #     num_nodes = features.size()[0]
    #     dim_len = features.size()[1]
    #     all_hyperedges = []
    #     num_clusters = 0
    #     self.num_combinations, self.subset_size = 300, 5
    #     self.silhouette_threshold = 0
    #     combinations = self.random_dimension_combinations(dim_len, self.num_combinations, self.subset_size)
        
    #     for combination in combinations:
    #         selected_features = features[:, combination]
    #         db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    #         labels = db.fit_predict(selected_features)

    #         unique_labels = set(labels)
    #         if -1 in unique_labels:
    #             unique_labels.remove(-1)  # Remove noise label if present
            
    #         if len(unique_labels) > 1:  # Ensure more than one cluster exists
    #             silhouette_avg = silhouette_score(selected_features, labels)
    #             # print(silhouette_avg)
    #             if silhouette_avg >= self.silhouette_threshold:
    #                 for label in unique_labels:
    #                     indices = np.where(labels == label)[0]
    #                     print(indices)
    #                     if len(indices) > 1:  # Ensure hyperedges have at least 2 nodes
    #                         hyperedge = indices.tolist()
    #                         if hyperedge not in all_hyperedges:
    #                             all_hyperedges.append(hyperedge)
    #                             num_clusters += 1

    #     if num_clusters > 0:  # Check if there are any clusters before proceeding
    #         incidence_matrix = self.make_incidence_matrix(all_hyperedges, num_nodes, num_clusters)
    #         return incidence_matrix
    #     else:
    #         print("No valid clusters found. Returning empty incidence matrix.")
    #         return np.zeros((num_nodes, 1))
        

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
        self.hidden_num = args.hidden_num
        self.num_layers = config.n_layers
        self.vc = VertexConv(args, self.dim_in, self.dim_in)
        self.construct = HGConstruct(args, config)
        self.ec = EdgeConv(args, self.dim_in, self.dim_in)
    
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
        self.args=args
        self.dim_in = args.dim_in
        self.n_categories = args.n_categories  # Number of output categories
        self.has_bias = True  # Whether to use bias in the fc layer
        self.dropout = nn.Dropout(args.dropout_rate)
        self.vc = VertexConv(args, self.dim_in, self.dim_in)
        self.ec = HypergraphEdgeAggregation(args, self.dim_in, self.dim_in)
        self.fc = nn.Linear(self.dim_in, self.n_categories, bias=self.has_bias)
        self.activation = nn.LogSoftmax(dim=-1)  # Activation function for the output layer
    
    def forward(self, node_feats, edge_feats, inc_mat):
        self.fc=self.fc.to(self.args.device)
        edge_feat = self.vc(node_feats, inc_mat, edge_feats)
        logits = self.ec(edge_feat)
        logits = self.fc((logits))
        return logits
    
class HGScanLayer(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.msg_pass = MessagePassing(args, config)
        self.conv = HGConv(args)
    
    def forward(self, inputs, inc_mat):
        node_feats, edge_feats, inc_mat = self.msg_pass(inputs, inc_mat)
        logits = self.conv(node_feats, edge_feats, inc_mat)
        return logits


        

