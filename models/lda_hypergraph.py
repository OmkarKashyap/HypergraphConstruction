import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import torch
import torch.nn as nn

import torch.nn.functional as F


class SemanticHypergraphModel(nn.Module):
    def __init__(self, args):
        super(SemanticHypergraphModel, self).__init__()
        self.num_topics = args.num_topics
        self.word_dimension = args.dim_in
        self.num_classes = args.n_categories
        self.top_k = args.top_k
        self.topic_vectors = nn.Parameter(torch.randn(self.num_topics, self.word_dimension))
        self.fc = nn.Linear(self.num_topics, self.num_classes)
        
    def forward(self, inputs):
        # inputs shape: [batch_size, max_len, word_dimension]
        batch_size, max_len, word_dimension = inputs.size()
        
        # Flatten inputs to [batch_size * max_len, word_dimension]
        flattened_inputs = inputs.view(batch_size * max_len, word_dimension)
        
        # Compute topic distributions using softmax
        topic_distributions = F.softmax(self.topic_vectors, dim=1)  # shape (num_topics, word_dimension)
        
        # Select top K words for each topic
        top_indices = torch.topk(topic_distributions, self.top_k, dim=1)[1]  # shape (num_topics, top_k)
        
        # Create masks for selected top words
        masks = torch.zeros_like(topic_distributions)  # shape (num_topics, word_dimension)
        for i in range(self.num_topics):
            masks[i, top_indices[i]] = 1.0
        
        # Apply masks to flattened_inputs
        masked_inputs = torch.matmul(masks, flattened_inputs.transpose(0, 1)).transpose(0, 1)
        
        # Reshape masked_inputs back to [batch_size, max_len, num_topics]
        masked_inputs = masked_inputs.view(batch_size, max_len, self.num_topics)
        
        # Sum along max_len dimension to get document representation
        document_representation = torch.sum(masked_inputs, dim=1)  # shape [batch_size, num_topics]
        
        # Apply fully connected layer
        output = self.fc(document_representation)  # shape [batch_size, num_classes]
        
        return output