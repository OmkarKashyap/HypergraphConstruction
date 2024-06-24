import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import torch
import torch.nn as nn

import torch.nn.functional as F

class TextProcessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, texts):
        processed_texts = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
            processed_texts.append(filtered_tokens)
        return processed_texts

class LDATopicModel(torch.nn.Module):
    def __init__(self, texts, num_topics=5, num_words=5):
        super(LDATopicModel, self).__init__()
        self.texts = texts
        self.num_topics = num_topics
        self.num_words = num_words
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.lda_model = models.LdaModel(self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=20)
        
    def get_topics(self):
        topics = self.lda_model.print_topics(num_topics=self.num_topics, num_words=self.num_words)
        hyperedges = []
        for topic in topics:
            words = [word for word, _ in self.lda_model.show_topic(topic[0])]
            hyperedges.append(words)
        return hyperedges


class SemanticHypergraphModel(nn.Module):
    def __init__(self, num_topics, word_dimension, num_classes, top_k):
        super(SemanticHypergraphModel, self).__init__()
        self.num_topics = num_topics
        self.word_dimension = word_dimension
        self.num_classes = num_classes
        self.top_k = top_k
        self.topic_vectors = nn.Parameter(torch.randn(num_topics, word_dimension))
        self.fc = nn.Linear(num_topics, num_classes)
        
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