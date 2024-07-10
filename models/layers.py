import sklearn
from sklearn.cluster import DBSCAN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.omk_dep_hg.dep_hg import DependencyHG
from models.lda_hypergraph import SemanticHypergraphModel

class HGConstruct(nn.Module):
    def __init__(self,args, config) -> None:
        super(HGConstruct, self). __init__()
        self.args=args
        self.eps = config.eps
        self.min_samples = config.min_samples
        self.clusters = None
        self.dep = DependencyHG(args, config)
        self.lda = SemanticHypergraphModel(args, config)
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
        lda_hg = self.lda(inputs).to(self.args.device)
        dep_hg = self.dep(inputs).to(self.args.device)
        inc_mat = inc_mat.to(self.args.device)
        final_inc = torch.cat((dep_hg, inc_mat, lda_hg), dim=2)
        return final_inc

class VertexConv(nn.Module):
    def __init__(self, args, config, in_feats, out_feats):
        super(VertexConv, self).__init__()
        self.args=args
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.num_heads = config.attention_heads
        self.dropout = config.dropout_rate

        #MultiHead attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim = in_feats, num_heads=self.num_heads, dropout = self.dropout)
        
        self.projection = nn.Linear(in_feats, out_feats, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

        #Alpha parameter for combining feats
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

        #Layer Normalization
        self.layer_norm = nn.LayerNorm(out_feats)

        #Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)


    def forward(self, feature_mat, inc_mat, prev_edge_features=None):
        # feature_mat: (m x d) -> m: number of nodes, d: input feature dimension
        # inc_mat: (m x e) -> m: number of nodes, e: number of edges

        b, m, d = feature_mat.size()
        b, _, e = inc_mat.size()
        self.projection=self.projection.to(self.args.device)

        #Permute feature_mat to (m x b x d) for multi head attention
        feature_mat = feature_mat.permute(1,0,2) # (m x b x d)

        attn_output, _ = self.multihead_attention(feature_mat, feature_mat, feature_mat) # (m x b x d)

        attn_output = attn_output.permute(1,0,2) # (b, m, d)

        # Compute attention scores for each node for each edge
        node_attention_scores = torch.bmm(inc_mat.transpose(1,2), attn_output) # (b, e, d)

        attention_weights = F.softmax(node_attention_scores, dim=1) # (b, e, d)

        # Compute weighted node features
        weighted_node_features = torch.einsum('bme,bmd->bmed', inc_mat, attn_output) # (b x m x e x d)

        # Aggregate weighted node features to get edge features
        edge_features = torch.einsum('bmed,bed->bed', weighted_node_features, attention_weights)  # (b x e x d)

        #Apply dropout
        edge_features = self.dropout_layer(edge_features)

        # Project aggregated edge features to the output dimension
        edge_features = self.projection(edge_features) # (b x e x out_feats)

        #Apply layer norm
        edge_features = self.layer_norm(edge_features.view(-1, self.out_feats)).view(edge_features.size()) #(b x e x out_feats)

        # If previous edge features are provided, combine them with the new ones based on alpha
        if prev_edge_features is not None:
            edge_features = self.alpha * prev_edge_features + (1 - self.alpha) * edge_features

        # Residual connection
        residual_edge_features = edge_features + prev_edge_features if prev_edge_features is not None else edge_features

        return residual_edge_features

class EdgeConv(nn.Module):
    def __init__(self, args, config, in_feats, out_feats):
        super(EdgeConv, self).__init__()
        self.args=args
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.num_heads = config.attention_heads
        self.dropout = config.dropout_rate

        #MultiHead attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim = in_feats, num_heads=self.num_heads, dropout = self.dropout)

        self.projection = nn.Linear(in_feats, out_feats, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

        #Layer Normalization
        self.layer_norm = nn.LayerNorm(out_feats)

        #Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, inc_mat, edge_features, prev_node_features=None):
        # inc_mat: (m x e) -> m: number of nodes, e: number of edges
        # edge_features: (e x d) -> e: number of edges, d: edge feature dimension

        b, m, e = inc_mat.size()
        self.projection=self.projection.to(self.args.device)

        edge_features = edge_features.permute(1,0,2) #(e x b x d)

        attn_output, _ = self.multihead_attention(edge_features, edge_features, edge_features) # (e x b x d)

        attn_output = attn_output.permute(1,0,2) # (b, e, d)

        edge_attention_scores = torch.einsum('bed, bme->bmd', attn_output, inc_mat) # (b x m x d)

        attention_weights = F.softmax(edge_attention_scores, dim=1) #( b x m x d)
        
        weighted_edge_features = torch.einsum('bed, bme->bmd', edge_features, inc_mat) # (b x m x d )

        #Apply Dropout
        weighted_edge_features = self.dropout_layer(weighted_edge_features)

        node_features = self.projection(weighted_edge_features)

        #Apply Layer Norm
        node_features = self.layer_norm(node_features.view(-1, self.out_feats)).view(node_features.size()) # (b x m x out_feats)

        # If previous node features are provided, combine them with the new ones based on alpha
        if prev_node_features is not None:
            node_features = self.alpha * prev_node_features + (1 - self.alpha) * node_features

        # Residual connection
        residual_node_features = node_features + prev_node_features if prev_node_features is not None else node_features

        return residual_node_features
        

class HypergraphEdgeAggregation(nn.Module):
    def __init__(self, args, config, in_features, out_features):
        super(HypergraphEdgeAggregation, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features

        self.num_heads = config.attention_heads
        self.dropout = config.dropout_rate

        #MultiHead attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim = self.in_features, num_heads=self.num_heads, dropout = self.dropout)

        self.projection = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

        #Batch Normalization 
        self.batch_norm = nn.BatchNorm1d(out_features)

        #dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, edge_features):
        """
        Perform edge aggregation for hypergraph representation using attention mechanism.
        
        :param edge_features: Aggregated node features for each edge (e x d)
        :return: Aggregated edge features (out_features)
        """

        self.projection=self.projection.to(self.args.device)
        
        edge_features = edge_features.permute(1,0,2) # (e x b x d)

        attn_output, _ = self.multihead_attention(edge_features, edge_features, edge_features)

        attn_output = attn_output.permute(1,0,2) # (b x e x d )

        #Sum over edges
        pooled_features = torch.sum(attn_output, dim=1) # (b x d)

        #Apply dropout
        output_features = self.dropout_layer(pooled_features) 

        output_features = self.projection(output_features) #(b x out_features)

        #Batch Norm
        output_features = self.batch_norm(output_features)
        
        return output_features


class MessagePassing(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.args=args
        self.dim_in = args.dim_in
        self.hidden_num = args.hidden_num
        self.num_layers = config.n_layers

        self.vc = VertexConv(args, config, self.dim_in, self.dim_in)
        self.construct = HGConstruct(args, config)
        self.ec = EdgeConv(args, config, self.dim_in, self.dim_in)

        for module in [self.vc, self.construct, self.ec]:
            for name,param in module.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

        self.dropout = nn.Dropout(self.args.dropout_rate)
    
    def forward(self, inputs, inc_mat):
        # features = features.squeeze(0)
        inc_mat = self.construct(inputs, inc_mat)
        features = inputs[0]
        edge_feats = self.vc(features, inc_mat)
        node_feats = self.ec(inc_mat, edge_feats, features)

        for _ in range(self.num_layers):
            
            edge_feats = self.vc(features, inc_mat, edge_feats)
            node_feats = self.ec(inc_mat, edge_feats, node_feats)

        #Apply dropout regularization
        node_feats = self.dropout(node_feats)
        
        return node_feats, edge_feats, inc_mat

class HGConv(nn.Module):
    def  __init__(self, args, config) -> None:
        super(). __init__()
        self.args=args
        self.dim_in = args.dim_in
        self.n_categories = args.n_categories  # Number of output categories
        self.has_bias = True  # Whether to use bias in the fc layer
        self.dropout = nn.Dropout(args.dropout_rate)
        self.vc = VertexConv(args, config, self.dim_in, self.dim_in)
        self.ec = HypergraphEdgeAggregation(args, config, self.dim_in, self.dim_in)

        self.fc = nn.Linear(self.dim_in, self.n_categories, bias=self.has_bias)
        nn.init.xavier_uniform_(self.fc.weight)

        self.activation = nn.LogSoftmax(dim=-1)  # Activation function for the output layer
    

    def forward(self, node_feats, edge_feats, inc_mat):

        node_feats = node_feats.to(self.args.device)
        edge_feats = edge_feats.to(self.args.device)
        inc_mat = inc_mat.to(self.args.device)
        self.fc=self.fc.to(self.args.device)

        
        edge_feat = self.vc(node_feats, inc_mat, edge_feats)
        edge_feat = self.dropout(edge_feat)

        logits = self.ec(edge_feat)
        logits = self.fc((logits))
        return logits
    
class HGScanLayer(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.msg_pass = MessagePassing(args, config)
        self.conv = HGConv(args, config)
    
    def forward(self, inputs, inc_mat):
        node_feats, edge_feats, inc_mat = self.msg_pass(inputs, inc_mat)
        logits = self.conv(node_feats, edge_feats, inc_mat)
        logits = self.activation(logits)
        return logits


        

