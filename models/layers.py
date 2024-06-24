import sklearn
from sklearn.cluster import DBSCAN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class HGConstruct(nn.Module):
    def __init__(self, eps, min_samples, output_size) -> None:
        super(HGConstruct, self).__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = None
        self.output_size = output_size
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
        num_nodes = features.size()[1]
        dim_len = features.size()[2]
        batch_incidence_matrix = []
       
        for batch in range(features.size()[0]):
            num_clusters = 0
            hyperedges = []

            for dim in range(dim_len):
                # print(features.squeeze(0)[:,dim].reshape(-1, 1).shape)\
                # reshaped_tensor = features.view(85, 1)
                # print(reshaped_tensor.shape)
                current_dim_emb = features[batch,:,dim].reshape(-1, 1)
                db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                labels = db.fit_predict(current_dim_emb)
                unique_labels = set(labels)
                unique_labels.remove(-1)
                clusters_now = len(unique_labels)
                dim_clusters = [np.where(labels == label)[0].tolist() for label in unique_labels]
                num_clusters += clusters_now
                if dim_clusters:
                    hyperedges.extend(dim_clusters)
        
            incidence_matrix = self.make_incidence_matrix(hyperedges, num_nodes, num_clusters)
            batch_incidence_matrix.append(incidence_matrix)
        batch_incidence_matrix = torch.stack(batch_incidence_matrix)
        return batch_incidence_matrix

    # def sentence_level(self, inputs):
    #     '''
    #     performs attention over all the tokens in the sentence.
    #     returns the feature vector for a sentence.
    #     :param inputs: Tensor of shape [batch_size, units, input_size]
    #                 `input_size` Dimensionality of each token's embedding
    #                 `units` Number of tokens in the sentence.
    #                 `batch_size` will be preserved
    #     '''
    #     input_size = inputs.size(dim=2)
    #     input_projection = nn.Linear(input_size, self.output_size)
    #     input_projection = torch.tanh(input_projection(inputs))
    #     vector_attn = torch.sum(input_projection * self.attention_context_vector, dim=2, keepdim=True)  # [batch_size, units, 1]
    #     attention_weights = F.softmax(vector_attn, dim=1)  # [batch_size, units, 1]
        
    #     # Compute the weighted projection
    #     weighted_projection = input_projection * attention_weights  # [batch_size, units, output_size]
        
    #     # Sum over the units axis to get the final output
    #     outputs = torch.sum(weighted_projection, dim=1)  # [batch_size, output_size]
        
    #     return outputs
    
class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, input_dim, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, input_dim, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats

class VertexConv(nn.Module):
    """
    A Vertex Convolution layer.
    Transform (N, m, d) feature to (N, e, d) feature by transform matrix and 1-D convolution.
    """
    def __init__(self, input_dim):
        """
        :param input_dim: input feature dimension.
        """
        super().__init__()

        self.input_dim = input_dim
        self.activation = nn.ReLU()
        self.dp = nn.Dropout()

    def forward(self, feature_matrix, incidence_matrix):
        """
        :param feature_matrix: (N, m, d) - batch of feature matrices.
        :param incidence_matrix: (N, m, e) - batch of incidence matrices.
        :return: (N, e, d) - batch of hyperedge feature matrices.
        """
        N, m, d = feature_matrix.size()
        _, _, e = incidence_matrix.size()
        self.convK1 = nn.Conv1d(m, 1, 1)  # (N, k, d) -> (N, 1, d)
        # Expand dimensions for broadcasting
        incidence_matrix = incidence_matrix.unsqueeze(-1)  # (N, m, e, 1)

        # Select node features that are part of each hyperedge
        hyperedge_node_feats = feature_matrix.unsqueeze(2) * incidence_matrix  # (N, m, 1, d) * (N, m, e, 1) -> (N, m, e, d)
        hyperedge_node_feats = hyperedge_node_feats.permute(0, 2, 1, 3)  # (N, e, m, d)
        self.trans = Transform(self.input_dim, m)  # (N, k, d) -> (N, k, d)
        # Transform features
        transformed_feats = self.activation(self.trans(hyperedge_node_feats))  # (N, e, m, d)
        transformed_feats = self.dp(transformed_feats)  # Apply dropout

        # Apply convolution and pool features
        transformed_feats = transformed_feats.view(N * e, m, d).transpose(1, 2)  # (N*e, d, m)
        pooled_feats = self.convK1(transformed_feats)  # (N*e, d, 1)
        pooled_feats = pooled_feats.squeeze(-1).view(N, e, d)  # (N, e, d)
        
        return pooled_feats


class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feats):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param feats (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = feats.size(1)
        for i in range(n_edges):
            scores.append(self.fc(feats[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        
        return (scores * feats).sum(1)

class HGScanLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.eps = 0.01
        self.min_samples = 3
        self.output_size = 10
        self.dim_in = 300
        self.hidden_num = 5
        self.ft_dim = 300
        self.n_categories = 3  # Number of output categories
        self.has_bias = True  # Whether to use bias in the fc layer
        self.dropout = nn.Dropout(0.5)
        self.vc = VertexConv(self.dim_in)
        self.ec = EdgeConv(self.ft_dim, self.hidden_num)
        self.construct = HGConstruct(eps=self.eps, min_samples = self.min_samples, output_size=self.output_size)
        self.fc = nn.Linear(self.hidden_num, self.n_categories, bias=self.has_bias)
        self.activation = nn.LogSoftmax(dim=-1)  # Activation function for the output layer
    
    def forward(self, features):
        inc_mat = self.construct.cluster(features)
        edge_feat = self.vc(features, inc_mat)
        logits = self.ec(edge_feat)
        logits = self.activation(self.fc(self.dropout(logits)))
        return logits
    
    
class SqueezeEmbedding(nn.Module):
    '''
    Squeeze sequence embedding length to the longest one in the batch
    '''
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
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
        


        

