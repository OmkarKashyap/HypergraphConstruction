import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNHG(nn.Module):
    def __init__(self, args, config):
        super(KNNHG, self).__init__()
        self.args = args
        
        self.k = config.top_k_knn  # number of nearest neighbors
        self.fc = nn.Linear(args.dim_in, args.n_categories)
        
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x_complete):
        print(len(x_complete))
        x, text, aspect, pos_mask, word_mask, aspect_post_start, aspect_post_end, plain_text, text_list = x_complete
        
        hypergraph = self.generate_knn_hypergraph(x)
        
        return hypergraph

    def generate_knn_hypergraph(self, embedded):
        # print(type(embedded))
        # print(embedded)
        # print(embedded[0])
        batch_size, seq_len, embed_dim = embedded.shape
        
        # Reshape for KNN
        flattened = embedded.view(-1, embed_dim)
        
        # Compute KNN
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(flattened.detach().cpu().numpy())
        distances, indices = nbrs.kneighbors(flattened.detach().cpu().numpy())
        
        # Create hypergraph
        hypergraph = torch.zeros(batch_size * seq_len, batch_size * seq_len)
        for i in range(batch_size * seq_len):
            hypergraph[i, indices[i]] = 1
        
        # Reshape back to batch form
        hypergraph = hypergraph.view(batch_size, seq_len, batch_size, seq_len)
        hypergraph = hypergraph.sum(dim=2)  # Combine hyperedges for each batch
        
        return hypergraph
    
