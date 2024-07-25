import sklearn
from sklearn.cluster import DBSCAN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.omk_dep_hg.dep_hg import DependencyHG
from models.lda_hypergraph import SemanticHypergraphModel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import math
# from data_utils.data_utils import AspectAttention

class HGConstruct(nn.Module):
    def __init__(self,args, config) -> None:
        super(HGConstruct, self). __init__()
        self.args=args
        self.min_samples = config.min_samples
        self.clusters = None
        self.dep = DependencyHG(args, config)
        self.lda = SemanticHypergraphModel(args, config)
        self.config = config
        self.silhouette_threshold = config.silhouette_threshold

    def adaptive_eps(self, X, min_eps=1e-4):
        if isinstance(X, torch.Tensor):
            X = X.detach().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
        distances, _ = nbrs.kneighbors(X)
        eps = np.percentile(distances[:, -1], 90)
        return max(eps, min_eps)

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

    def dim_cluster(self, features):
        '''
        perform DBSCAN over the nodes, one dimension at a time.
        return an incidence matrix with each cluster as a hyperedge.
        :param ids: indices selected during train/valid/test, torch.LongTensor.
        :param features: assumed it is a tensor. (N, k, d)
        '''
        # print(features.shape)
        num_nodes = features.size()[0]
        dim_len = features.size()[1]
        num_clusters = 0
        hyperedges = []

        for dim in range(dim_len):
            # print(features.squeeze(0)[:,dim].reshape(-1, 1).shape)\
            # reshaped_tensor = features.view(85, 1)
            # print(reshaped_tensor.shape)
            dim_clusters = []
            current_dim_emb = features.squeeze(0)[:,dim].reshape(-1, 1).detach()
            # print(current_dim_emb.shape)
            eps = self.adaptive_eps(current_dim_emb)
            # print(eps)
            db = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = db.fit_predict(current_dim_emb)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            if len(unique_labels) > 1:
                silhouette_avg = silhouette_score(current_dim_emb, labels)
                # print("Silhouette Average:", silhouette_avg)
                if silhouette_avg >= self.silhouette_threshold:
                    dim_clusters = [np.where(labels == label)[0].tolist() for label in unique_labels]
                    
                if dim_clusters:
                    num_clusters += len(dim_clusters)
                    hyperedges.extend(dim_clusters)
                
        print("num_clusters", num_clusters)
        if not num_clusters:
            return torch.randint(low=0, high=2, size=(num_nodes, 40), dtype=torch.float)
        incidence_matrix = self.make_incidence_matrix(hyperedges, num_nodes, num_clusters)
    # batch_incidence_matrix = torch.stack(batch_incidence_matrix)
        return incidence_matrix

    def hier_cluster(self, embeddings):
        features = embeddings.numpy()

        # Perform hierarchical clustering
        Z = linkage(features, method='ward')  # Using 'ward' linkage method

        # Determine clusters automatically based on distance
        distance_threshold = 15  # Adjust as needed based on your data and clustering requirements
        clusters = fcluster(Z, t=distance_threshold, criterion='distance')
        num_clusters = len(np.unique(clusters))
        hyperedges = [[] for _ in range(num_clusters)]
        for idx, cluster_id in enumerate(clusters):
            hyperedges[cluster_id - 1].append(idx)
        hyperedges_updated = [hyperedge for hyperedge in hyperedges if len(hyperedge) > 1]

        incidence_matrix = self.make_incidence_matrix(hyperedges_updated, embeddings.size(0), len(hyperedges_updated))
        return incidence_matrix

    def forward(self, inputs, inc_mat):
        if self.config.model == 'dbscan':
            return inc_mat
        elif self.config.model == 'lda':
            lda_hg = self.lda(inputs).to(self.args.device)
            return lda_hg
        elif self.config.model == 'dephg':
            dep_hg = self.dep(inputs).to(self.args.device)
            return dep_hg
        else:
            lda_hg = self.lda(inputs).to(self.args.device)
            dep_hg = self.dep(inputs).to(self.args.device)
            inc_mat = inc_mat.to(self.args.device)
            final_inc = torch.cat((dep_hg, inc_mat, lda_hg), dim=2)
            return final_inc

class VertexConv(nn.Module):
    def __init__(self, args, in_feats, out_feats):
        super(VertexConv, self).__init__()
        self.args=args
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
        self.attention = nn.Linear(in_feats, out_feats, bias=False)
        self.projection = nn.Linear(in_feats, out_feats, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

    def forward(self, inc_mat, edge_features, prev_node_features=None):
        # inc_mat: (m x e) -> m: number of nodes, e: number of edges
        # edge_features: (e x d) -> e: number of edges, d: edge feature dimension

        b, m, e = inc_mat.size()

        self.attention = self.attention.to(self.args.device)
        self.projection=self.projection.to(self.args.device)
        edge_features = edge_features.to(self.args.device)
        # Compute attention scores for each edge
        edge_attention_scores = self.attention(edge_features)  # (e x out_feats)

        # Aggregate edge attention scores to get node attention scores
        # node_attention_scores = inc_mat @ edge_attention_scores  
        node_attention_scores = torch.bmm(inc_mat, edge_attention_scores) # (m x out_feats)

        # Apply softmax to get attention weights for each node
        attention_weights = F.softmax(node_attention_scores, dim=1)  # (m x out_feats)

        # Weighted aggregation of edge features to get new node features
        weighted_edge_features = torch.einsum('bed,bme->bmed', edge_features, inc_mat)  # (b x m x e x d)

        # Aggregate weighted edge features to get node features
        edge_features = torch.einsum('bmed,bmd->bmd', weighted_edge_features, attention_weights)  # (b x m x d)

        # Project aggregated node features to the output dimension
        node_features = self.projection(edge_features)  # (b x m x out_feats)
        node_features = node_features.to(self.args.device)
        # If previous node features are provided, combine them with the new ones based on alpha
        if prev_node_features is not None:
            prev_node_features = prev_node_features.to(self.args.device)
            # print("Node:", node_features.shape)
            # print("Previous Node:", prev_node_features.shape)
            node_features = self.alpha * prev_node_features + (1 - self.alpha) * node_features

        return node_features

class AspectAttention(nn.Module):
    def __init__(self, args, hidden_size):
        super(AspectAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.args = args

    def forward(self, token_embeddings, aspect_embedding):
        # token_embeddings: (seq_len, hidden_size)
        # aspect_embedding: (1, hidden_size)
        print(token_embeddings.shape)
        print(aspect_embedding.shape)
        seq_len = token_embeddings.size(1)

        # Repeat aspect_embedding seq_len times
        aspect_embedding = aspect_embedding.float()
        aspect_embedding = aspect_embedding.mean(dim=1, keepdim=True)  # (1, hidden_size)
        print("aspect_embedding.shape", aspect_embedding.shape)
        aspect_embedding = aspect_embedding.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate token embeddings with aspect embeddings
        combined = torch.cat((token_embeddings, aspect_embedding), dim=2).float()  # (batch_size, seq_len, hidden_size * 2)

        # Compute attention scores
        self.attention = self.attention.to(self.args.device)
        energy = torch.tanh(self.attention(combined))  # (batch_size, seq_len, hidden_size)
        energy = energy.transpose(1, 2)  # (batch_size, hidden_size, seq_len)

        # Compute v * energy
        v = self.v.repeat(token_embeddings.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attention_scores = torch.bmm(v, energy).squeeze(1)  # (batch_size, seq_len)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Compute weighted sum of token embeddings
        attention_weights = attention_weights.unsqueeze(2)  # (batch_size, seq_len, 1)
        print("attention_weights",attention_weights)

        aspect_aware_embeddings = token_embeddings * attention_weights  # (batch_size, seq_len, hidden_size)

        aspect_aware_embeddings = aspect_aware_embeddings + token_embeddings  # (batch_size, seq_len, hidden_size)

        return aspect_aware_embeddings
    
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
        self.num_layers = args.n_layers
        self.args = args
        # self.pos_encoder = HypergraphPositionalEncoding(args, args.dim_in, args.max_length)
        self.aspect_attention = nn.MultiheadAttention(args.dim_in, 1, batch_first=True)
        self.vc = VertexConv(args, self.dim_in, self.dim_in)
        self.construct = HGConstruct(args, config)
        self.ec = EdgeConv(args, self.dim_in, self.dim_in)
    
    def forward(self, inputs, inc_mat, aspect_emb):
        # features = features.squeeze(0)
        inc_mat = self.construct(inputs, inc_mat)
        features = inputs[0]
        self.aspect_attention = self.aspect_attention.to(self.args.device)
        print(features.shape)
        features, _ = self.aspect_attention(features, aspect_emb, aspect_emb)
        print(features.shape)
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
        self.args = args
        self.config = config
        self.msg_pass = MessagePassing(args, config)
        self.conv = HGConv(args)
        self.lstm = nn.LSTM(args.dim_in, args.dim_in, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(args.dim_in * 2, args.dim_in)
    
    def forward(self, inputs, inc_mat, aspect_emb):
        aspect_emb = aspect_emb.float()
        feats = inputs[0]
        print("Feats:", feats.shape)
        if self.config.seq:
            states, _ = self.lstm(feats)
            states = torch.stack(list(states)).to(self.args.device)
            self.lstm_proj = self.lstm_proj.to(self.args.device)
            feats_lstm = self.lstm_proj(states)
            feats = feats + feats_lstm
            inputs[0] = feats
        node_feats, edge_feats, inc_mat = self.msg_pass(inputs, inc_mat, aspect_emb)
        logits = self.conv(node_feats, edge_feats, inc_mat)
        return logits


    



# class HypergraphPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len):
#         super(HypergraphPositionalEncoding, self).__init__()
#         self.d_model = d_model
        
#         # Create base positional encodings
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # handle odd d_model
        
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x, inc_mat):
#         batch_size, num_nodes, _ = x.size()
        
#         # Calculate node degrees
#         node_degrees = inc_mat.sum(dim=-1).long()
        
#         # Ensure node degrees are within bounds
#         node_degrees = torch.clamp(node_degrees, max=self.pe.size(1) - 1)
        
#         # Get positional encodings for each node based on its degree
#         pos_encodings = self.pe[:, node_degrees]
        
#         # Ensure positional encodings have the correct shape
#         pos_encodings = pos_encodings.view(batch_size, num_nodes, self.d_model)
        
#         # Add positional encodings to input
#         return x + pos_encoding

# class EdgeConv(nn.Module):
#     def __init__(self, args, in_feats, out_feats):
#         super(EdgeConv, self).__init__()
#         self.args=args
#         self.in_feats = in_feats
#         self.out_feats = out_feats
#         self.attention = nn.Linear(in_feats, out_feats, bias=False)
#         self.projection = nn.Linear(in_feats, out_feats, bias=False)
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a trainable parameter

#     def forward(self, inc_mat, edge_features, prev_node_features=None):
#         # inc_mat: (m x e) -> m: number of nodes, e: number of edges
#         # edge_features: (e x d) -> e: number of edges, d: edge feature dimension

#         b, m, e = inc_mat.size()

#         self.attention = self.attention.to(self.args.device)
#         self.projection=self.projection.to(self.args.device)

#         # Compute attention scores for each edge
#         edge_attention_scores = self.attention(edge_features)  # (e x out_feats)

#         # Aggregate edge attention scores to get node attention scores
#         # node_attention_scores = inc_mat @ edge_attention_scores  
#         node_attention_scores = torch.matmul(inc_mat, edge_attention_scores) # (m x out_feats)

#         # Apply softmax to get attention weights for each node
#         attention_weights = F.softmax(node_attention_scores, dim=1)  # (m x out_feats)

#         # Weighted aggregation of edge features to get new node features
#         weighted_edge_features = torch.einsum('bed,bme->bmd', edge_features, inc_mat)  # (m x d)

#         # Project aggregated node features to the output dimension
#         node_features = self.projection(weighted_edge_features)  # (m x out_feats)

#         # If previous node features are provided, combine them with the new ones based on alpha
#         if prev_node_features is not None:
#             node_features = self.alpha * prev_node_features + (1 - self.alpha) * node_features

#         return node_features