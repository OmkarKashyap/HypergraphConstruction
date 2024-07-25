import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import torch
import torch.nn as nn

import torch.nn.functional as F


class SemanticHypergraphModel(nn.Module):
    def __init__(self, args, config):
        super(SemanticHypergraphModel, self).__init__()
        self.num_topics = config.num_topics
        self.word_dimension = args.dim_in
        self.num_classes = args.n_categories
        self.top_k = config.top_k
        self.topic_vectors = nn.Parameter(torch.randn(self.num_topics, self.word_dimension))
        self.fc = nn.Linear(self.num_topics, self.num_classes)
        
    def forward(self, inputs):
        inputs = inputs[0]
        # inputs shape: [batch_size, max_len, word_dimension]
        batch_size, max_len, word_dimension = inputs.shape
        
        # Flatten inputs to [batch_size * max_len, word_dimension]
        flattened_inputs = inputs.view(batch_size * max_len, word_dimension)
        
        # Compute topic distributions using softmax
        topic_distributions = F.softmax(self.topic_vectors, dim=1)  # shape (num_topics, word_dimension)
        
        # Select top K words for each topic
        top_indices = torch.topk(topic_distributions, self.top_k, dim=1)[1]  # shape (num_topics, top_k)
        
        top_indices = top_indices % max_len

        hypergraph = torch.zeros(batch_size,max_len, self.num_topics, device=inputs.device)

        for i in range(self.num_topics):
            for j in range(self.top_k):
                word_idx = top_indices[i,j]
                for b in range(batch_size):
                    hypergraph[b,word_idx,i]=1
        
        max_num_hyperedges = self.num_topics
        hypergraph = hypergraph.view(batch_size, max_len, max_num_hyperedges)
        return hypergraph