import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

import igraph as ig
import matplotlib.pyplot as plt
import itertools
from nltk.corpus import stopwords
import string
from community import community_louvain

# class Graph(object):
#     def __init__(self, args):
#         self.graph = nx.DiGraph()       
    
#     def add_node(self, node, **attrs):
#         self.graph.add_node(node, **attrs)
    
#     def add_edge(self, src, dest, **attrs):
#         self.graph.add_edge(src, dest, **attrs)
    
#     def get_adjacency_matrix(self):
#         return nx.adjacency_matrix(self.graph)
    
class Graph:
    def __init__(self, args):
        self.nodes = {}
        self.edges = {}
        self.args = args

    def add_node(self, idx, **attributes):
        self.nodes[idx] = attributes

    def add_edge(self, src, dst, **attributes):
        if src not in self.edges:
            self.edges[src] = []
        self.edges[src].append((dst, attributes))
        
class DependencyGraph(nn.Module):
    def __init__(self, args):
        super(DependencyGraph, self).__init__()
        self.args = args
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
    
    def remove_stopwords_and_punctuations(self, tokens):
        clean_tokens = [token for token in tokens if token.lower() not in self.stop_words and token not in self.punctuation]
        return clean_tokens
        
    def forward(self, inputs):
        feats, tokens, aspect, pos, post, head, deprel, sen_len, adj, pos_mask, word_mask, aspect_pos_start, aspect_pos_end,plain_text, text_list = inputs
            
        #shape of inputs : [batch_size, max_length, dim_in]
        batch_size = feats.shape[0]
        max_length = feats.shape[1]
        adj_matrices = torch.Tensor()
        
        for i in range(batch_size):
            graph = self.construct_graph(tokens[i], pos[i], head[i], deprel[i], sen_len[i], aspect_pos_start[i], aspect_pos_end[i])
            adj_matrix = self.graph_to_adj(graph)
            adj_matrices = torch.cat((adj_matrices, adj_matrix.unsqueeze(0)), dim=0)
        return adj_matrices
    
    
    def construct_graph(self, tokens, pos, head, deprel, sen_len, aspect_pos_start, aspect_pos_end):
        graph = Graph(self.args)
    
        for idx in range(sen_len):
            is_aspect = aspect_pos_start <= idx <= aspect_pos_end
            node_attributes = {
                'token': tokens[idx],
                'pos': pos[idx],
                'deprel': deprel[idx],
                'is_aspect': is_aspect
            }
            graph.add_node(idx, **node_attributes)
        
            if head[idx] != -1 and head[idx] != 0:  # if head exists
                # edge_attributes = {
                #     'deprel': deprel[idx]
                # }
                # graph.add_edge(head[idx] - 1, idx, **edge_attributes)
                graph.add_edge(head[idx] - 1, idx)
        return graph
    
    def graph_to_adj(self, graph):
        max_length = self.args.max_length
        adj_matrix = torch.zeros((max_length, max_length))
        
        # Iterate over edges in the graph
        for src, dest in graph.edges.items():
            
            # Get source and destination indices
            src_idx = src.item() if isinstance(src, torch.Tensor) else src
            dest_idx = dest[0][0].item() if isinstance(dest, torch.Tensor) else dest[0][0]
            
            # Populate adjacency matrix
            adj_matrix[src_idx, dest_idx] = 1.0

        return adj_matrix
    
    # def graph_to_adj(self, graph):
    #     batch_size = self.args.batch_size
    #     max_length = self.args.max_length
        
    #     # adj_matrices = torch.full((batch_size, max_length, max_length), -1.0)
    #     adj_matrices = torch.zeros((max_length, max_length))
        
    #     for i in range(batch_size):
            
    #         for edge in graph.graph.edges:
    #             src, dest = edge
    #             adj_matrices[src, dest] = 1.0
        
    #     return adj_matrices
    

# class GNNLayer(nn.Module):
#     def __init__(self, args, config):
#         super(GNNLayer, self).__init__()
#         self.args = args
#         self.aggregation_type = config.gnn_aggregation_type
#         self.linear = nn.Linear(args.dim_in, args.hidden_dim)
#         self.linear2 = nn.Linear(self.args.hidden_dim, self.args.dim_in)
        
#         if self.aggregation_type == 'attention':
#             self.attention = nn.Linear(args.hidden_dim, 1)
    
#     def forward(self, feats, adj, word_mask):
#         batch_size, max_length, _ = feats.size()

#         feats = feats.to(self.args.device)
#         adj = adj.to(self.args.device)
#         word_mask = torch.stack(word_mask).to(self.args.device)
        
#         # Ensure the linear layers are on the same device
#         self.linear.to(self.args.device)
#         self.linear2.to(self.args.device)

#         # Apply linear transformation to node features
#         feats_transformed = self.linear(feats)
        
#         if self.aggregation_type == 'sum':
#             # Sum Aggregation
#             feats_aggregated = torch.bmm(adj, feats_transformed)
        
#         elif self.aggregation_type == 'mean':
#             # Mean Aggregation
#             degree_matrix = adj.sum(dim=-1, keepdim=True).clamp(min=1)
#             feats_aggregated = torch.bmm(adj, feats_transformed) / degree_matrix
        
#         elif self.aggregation_type == 'max':
#             # Max Pooling Aggregation
#             feats_transformed = feats_transformed.unsqueeze(1).expand(-1, max_length, -1, -1)
#             adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, feats_transformed.size(-1))
#             feats_aggregated = (feats_transformed * adj_expanded).max(dim=2)[0]
        
#         elif self.aggregation_type == 'attention':
#             # Attention Mechanism
#             attn_weights = self.attention(feats_transformed).squeeze(-1)
#             attn_weights = F.softmax(attn_weights.masked_fill(word_mask == 0, -1e9), dim=-1)
#             attn_weights = attn_weights.unsqueeze(-1)
#             feats_aggregated = torch.bmm(adj * attn_weights, feats_transformed)
        
#         # Apply masking to ignore padded nodes
#         feats_aggregated = feats_aggregated * word_mask.unsqueeze(2)
        
#         # Apply non-linearity (e.g., ReLU)
#         feats_aggregated = torch.relu(feats_aggregated)
        
#         # Apply the second linear transformation
#         feats_out = self.linear2(feats_aggregated)
        
#         return feats_out

class GNNLayer(nn.Module):
    def __init__(self, args, config):
        super(GNNLayer, self).__init__()
        self.args = args
        self.aggregation_type = config.gnn_aggregation_type

        self.linear = nn.Linear(args.dim_in, args.hidden_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

        self.linear2 = nn.Linear(self.args.hidden_dim, self.args.dim_in)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.norm_ln = nn.LayerNorm(args.hidden_dim)
        self.residual=True
        
        if self.aggregation_type == 'attention':
            self.attention_heads = config.attention_heads
            self.attention = nn.ModuleList([nn.Linear(args.hidden_dim, args.hidden_dim) for _ in range(self.attention_heads)])
            self.attention_weights = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(self.attention_heads)])

            for attn_layer in self.attention:
                nn.init.xavier_normal_(attn_layer.weight)
                nn.init.constant_(attn_layer.bias, 0)

            for attn_weight in self.attention:
                nn.init.xavier_normal_(attn_weight.weight)
                nn.init.constant_(attn_weight.bias, 0)

        if self.residual:
            self.res_linear = nn.Linear(args.dim_in, args.hidden_dim)
            nn.init.xavier_normal_(self.res_linear.weight)
            nn.init.constant_(self.res_linear.bias, 0)

            self.res_norm = nn.LayerNorm(args.hidden_dim)
        
    
    def forward(self, feats, adj, word_mask):
        batch_size, max_length, _ = feats.size()

        feats = feats.to(self.args.device)
        adj = adj.to(self.args.device)
        word_mask = torch.stack(word_mask).to(self.args.device)
        
        # Ensure the linear layers are on the same device
        self.linear.to(self.args.device)
        self.linear2.to(self.args.device)

        # Apply linear transformation to node features
        feats_transformed = self.linear(feats)
        
        if self.aggregation_type == 'sum':
            # Sum Aggregation
            feats_aggregated = torch.bmm(adj, feats_transformed)
        
        elif self.aggregation_type == 'mean':
            # Mean Aggregation
            degree_matrix = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            feats_aggregated = torch.bmm(adj, feats_transformed) / degree_matrix
        
        elif self.aggregation_type == 'max':
            # Max Pooling Aggregation
            feats_transformed = feats_transformed.unsqueeze(1).expand(-1, max_length, -1, -1)
            adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, feats_transformed.size(-1))
            feats_aggregated = (feats_transformed * adj_expanded).max(dim=2)[0]
        
        elif self.aggregation_type == 'attention':
            
            #Scaled Dot Product Attention Mechanism
            multi_head_outputs = []
            for attention, attention_weight in zip(self.attention, self.attention_weights):
                q = attention(feats_transformed)
                k = attention(feats_transformed)
                v = attention(feats_transformed)

                attn_scores = torch.bmm(q, k.transpose(1,2)) / (self.args.hidden_dim ** 0.5)
                attn_scores = attn_scores.masked_fill(word_mask.unsqueeze(1).eq(0), -1e9)
                attn_weights = F.softmax(attn_scores, dim=-1)

                feats_aggregated_head = torch.bmm(attn_weights, v)
                multi_head_outputs.append(feats_aggregated_head)

            feats_aggregated = torch.cat(multi_head_outputs, dim=-1)

            #Apply mask to ignore pad tokens
            feats_aggregated = feats_aggregated * word_mask.unsqueeze(2)

            #Apply Layer Norm
            feats_aggregated = self.norm_ln(feats_aggregated)

            #Apply non linearity and dropout
            feats_aggregated = F.relu(feats_aggregated)
            feats_aggregated = self.dropout(feats_aggregated)

            #Apply second linear layer
            feats_out = self.linear2(feats_aggregated)

            #Apply residual connection
            if self.residual:
                res_feats = self.res_linear(feats)
                res_feats = self.res_norm(res_feats)
                feats_out += res_feats
            
            return feats_out
    
class GCN(nn.Module):
    def __init__(self,args, config):
        super(GCN, self).__init__()
        self.args= args
        self.num_gnn_layers = config.num_gnn_layers

        self.gnn_layers = nn.ModuleList([GNNLayer(args,config) for _ in range(self.num_gnn_layers)])

    def forward(self, feats, adj, word_mask):
        out=feats

        for layer in self.gnn_layers:
            out = layer(out,adj, word_mask)
        return out

class CommunityDetection(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def louvain(self, adj, node_embeddings, tokens, text_list, word_mask):
        batch_size, max_length, _ = node_embeddings.size()
        incidence_matrices = []
        max_communities = 0
        community_list = []

        for i in range(batch_size):
            G = nx.Graph()
            for j in range(max_length):
                if word_mask[i][j] == 0:
                    continue
                G.add_node(j, feature=node_embeddings[i, j].cpu().detach().numpy())
           
            # Add edges to the graph based on the adjacency matrix
            adj_np = adj[i].cpu().detach().numpy()  # Convert adjacency matrix to numpy
            for u in range(max_length):
                for v in range(u + 1, max_length):
                    if adj_np[u, v] > 0:
                        G.add_edge(u, v, weight=adj_np[u, v])
           
            # Perform Louvain community detection
            partition = community_louvain.best_partition(G)
            communities = {}
            for node, community in partition.items():
                if community not in communities:
                    communities[community] = []
                communities[community].append(node)
            community_list.append(communities)  
            max_communities = max(max_communities, len(communities))
       
        # Construct and pad incidence matrices
        for i, communities in enumerate(community_list):
            incidence_matrix = np.zeros((max_length, max_communities))
            for hyperedge_id, nodes in communities.items():
                for node in nodes:
                    incidence_matrix[node, hyperedge_id] = 1
            incidence_matrices.append(torch.tensor(incidence_matrix, dtype=torch.float32))
       
        incidence_matrices = torch.stack(incidence_matrices)
        return community_list, incidence_matrices
        
    
    def girvan_newman(self, embeddings, tokens_batch, text_list, word_mask, min_degree=2):
        """
        Perform Girvan-Newman community detection on the provided embeddings after removing padding tokens.

        Args:
        - embeddings (torch.Tensor): Node embeddings of shape (batch_size, num_nodes, embedding_dim).
        - tokens_batch (list of lists): List of tokens for each graph in the batch.
        - text_list (list of str): List of text samples or document identifiers (for reference).
        - word_mask (torch.Tensor): Mask indicating padding tokens of shape (batch_size, max_length).

        Returns:
        - List of lists containing community assignments for each node in each graph in the batch.
        - Tensor of padded incidence matrices for each graph in the batch, structured as [batch_size, max_length, max_number of communities].
        """
        self.embeddings = embeddings
        self.threshold = self.args.girvan_newman_threshold
        batch_size, num_nodes, embedding_dim = embeddings.size()
        
        communities_batch = []
        incidence_matrices_batch = []
        max_num_hyperedges = 0
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust rows and cols based on batch size
        axes = axes.flatten()

        for batch_idx in range(batch_size):
            tokens = tokens_batch[batch_idx]
    
            # Get non-padding token indices
            non_padding_indices = torch.nonzero(word_mask[batch_idx]).squeeze()
            tokens = [tokens[i.item()] for i in non_padding_indices]
            num_tokens = len(tokens)
            
            # Extract embeddings and adjacency matrix for non-padding tokens
            embeddings_batch = embeddings[batch_idx, non_padding_indices]
            
            adj_matrix = torch.mm(embeddings_batch, embeddings_batch.transpose(0, 1))
            
            # Apply thresholding to convert to binary adjacency matrix
            adj_matrix_binary = (adj_matrix > self.threshold).float().to(self.args.device)
            
            # Create padded adjacency matrix
            padded_adj_matrix = torch.full((self.args.max_length, self.args.max_length), -1.0).to(self.args.device)
            padded_adj_matrix[non_padding_indices[:, None], non_padding_indices] = adj_matrix_binary
            
            # Convert adjacency matrix to networkx graph
            G = nx.DiGraph()
            max_length = padded_adj_matrix.shape[0]
            
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if adj_matrix_binary[i, j] > 0:
                        G.add_edge(i, j)
            
            print(f"Graph for {text_list[batch_idx]}:")
            print("Number of nodes:", G.number_of_nodes())
            print("Number of edges:", G.number_of_edges())
        
            # Perform Girvan-Newman community detection
            comp = nx.algorithms.community.girvan_newman(G)
            limited = itertools.takewhile(lambda c: len(c) <= 10, comp)
            communities = next(limited)

            # Convert communities to a list of sets
            hyperedges = list(communities)

            # Determine number of nodes and hyperedges
            num_nodes = max_length
            num_hyperedges = len(hyperedges)
            
            # Update max_num_hyperedges if current graph has more hyperedges
            if num_hyperedges > max_num_hyperedges:
                max_num_hyperedges = num_hyperedges

            # Initialize incidence matrix
            incidence_matrix = np.full((num_nodes, num_hyperedges), -1)

            # Populate incidence matrix based on hyperedges
            for h_idx, hyperedge in enumerate(hyperedges):
                for node_idx in hyperedge:
                    incidence_matrix[node_idx, h_idx] = 1

            # Append results to batch lists
            communities_batch.append(communities)
            incidence_matrices_batch.append(incidence_matrix)
            
            # Visualize the graph with communities
            ax = axes[batch_idx]
            pos = nx.spring_layout(G)
            
            # Assign a unique color to each community
            colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
            for community in communities:
                color = next(colors)
                nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=500, alpha=0.8, ax=ax)
            
            nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
            nx.draw_networkx_labels(G, pos, labels={i: tokens[i] for i in range(num_tokens)}, font_size=10, ax=ax)
            
            ax.set_title(f"Graph for {text_list[batch_idx]}")
        
        plt.tight_layout()
        plt.show()

        # Pad incidence matrices to the max number of hyperedges and convert to tensor
        padded_incidence_matrices_batch = []
        for incidence_matrix in incidence_matrices_batch:
            padded_incidence_matrix = np.pad(incidence_matrix, ((0, 0), (0, max_num_hyperedges - incidence_matrix.shape[1])), mode='constant', constant_values=-1)
            padded_incidence_matrices_batch.append(torch.tensor(padded_incidence_matrix, dtype=torch.float32))

        # Stack all incidence matrices to get the final tensor of shape [batch_size, max_length, max_num_hyperedges]
        padded_incidence_matrices_batch = torch.stack(padded_incidence_matrices_batch)

        return communities_batch, padded_incidence_matrices_batch
    
    def label_propagation(self, embeddings, tokens_batch, text_list, word_mask, min_degree=2):
        """
        Perform label propagation community detection on the provided embeddings after removing padding tokens.

        Args:
        - embeddings (torch.Tensor): Node embeddings of shape (batch_size, num_nodes, embedding_dim).
        - tokens_batch (list of lists): List of tokens for each graph in the batch.
        - text_list (list of str): List of text samples or document identifiers (for reference).
        - word_mask (torch.Tensor): Mask indicating padding tokens of shape (batch_size, max_length).

        Returns:
        - List of lists containing community assignments for each node in each graph in the batch.
        - Tensor of padded incidence matrices for each graph in the batch, structured as [batch_size, max_length, max_number of communities].
        """
        self.embeddings = embeddings
        self.threshold = self.args.label_propogation_threshold
        batch_size, num_nodes, embedding_dim = embeddings.size()
        
        communities_batch = []
        incidence_matrices_batch = []
        max_num_hyperedges = 0
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust rows and cols based on batch size
        axes = axes.flatten()

        for batch_idx in range(batch_size):
            tokens = tokens_batch[batch_idx]
    
            # Get non-padding token indices
            non_padding_indices = torch.nonzero(word_mask[batch_idx]).squeeze()
            tokens = [tokens[i.item()] for i in non_padding_indices]
            num_tokens = len(tokens)
            
            # Extract embeddings and adjacency matrix for non-padding tokens
            embeddings_batch = embeddings[batch_idx, non_padding_indices]
            
            adj_matrix = torch.mm(embeddings_batch, embeddings_batch.transpose(0, 1))
            
            # Apply thresholding to convert to binary adjacency matrix
            adj_matrix_binary = (adj_matrix > self.threshold).float()
            
            # Create padded adjacency matrix
            padded_adj_matrix = torch.full((self.args.max_length, self.args.max_length), -1.0)
            padded_adj_matrix[non_padding_indices[:, None], non_padding_indices] = adj_matrix_binary
            
            # Convert adjacency matrix to networkx graph
            G = nx.Graph()
            max_length = padded_adj_matrix.shape[0]
            
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if adj_matrix_binary[i, j] > 0:
                        G.add_edge(i, j)
            
            print(f"Graph for {text_list[batch_idx]}:")
            print("Number of nodes:", G.number_of_nodes())
            print("Number of edges:", G.number_of_edges())
        
            # Perform label propagation community detection
            communities = nx.algorithms.community.label_propagation_communities(G)

            # Convert communities to a list of sets
            hyperedges = list(communities)

            # Determine number of nodes and hyperedges
            num_nodes = max_length
            num_hyperedges = len(hyperedges)
            
            # Update max_num_hyperedges if current graph has more hyperedges
            if num_hyperedges > max_num_hyperedges:
                max_num_hyperedges = num_hyperedges

            # Initialize incidence matrix
            incidence_matrix = np.full((num_nodes, num_hyperedges), -1)

            # Populate incidence matrix based on hyperedges
            for h_idx, hyperedge in enumerate(hyperedges):
                for node_idx in hyperedge:
                    incidence_matrix[node_idx, h_idx] = 1

            # Append results to batch lists
            communities_batch.append(communities)
            incidence_matrices_batch.append(incidence_matrix)
            
            # Visualize the graph with communities
            ax = axes[batch_idx]
            pos = nx.spring_layout(G)
            
            # Assign a unique color to each community
            colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
            for community in communities:
                color = next(colors)
                nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=500, alpha=0.8, ax=ax)
            
            nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
            nx.draw_networkx_labels(G, pos, labels={i: tokens[i] for i in range(num_tokens)}, font_size=10, ax=ax)
            
            ax.set_title(f"Graph for {text_list[batch_idx]}")
        
        plt.tight_layout()
        plt.show()

        # Pad incidence matrices to the max number of hyperedges and convert to tensor
        padded_incidence_matrices_batch = []
        for incidence_matrix in incidence_matrices_batch:
            padded_incidence_matrix = np.pad(incidence_matrix, ((0, 0), (0, max_num_hyperedges - incidence_matrix.shape[1])), mode='constant', constant_values=-1)
            padded_incidence_matrices_batch.append(torch.tensor(padded_incidence_matrix, dtype=torch.float32))

        # Stack all incidence matrices to get the final tensor of shape [batch_size, max_length, max_num_hyperedges]
        padded_incidence_matrices_batch = torch.stack(padded_incidence_matrices_batch)

        return communities_batch, padded_incidence_matrices_batch
    
    def kernighan_lin(self, embeddings, tokens_batch, text_list, word_mask, min_degree=2):
        """
        Perform Kernighan-Lin community detection on the provided embeddings after removing padding tokens.

        Args:
        - embeddings (torch.Tensor): Node embeddings of shape (batch_size, num_nodes, embedding_dim).
        - tokens_batch (list of lists): List of tokens for each graph in the batch.
        - text_list (list of str): List of text samples or document identifiers (for reference).
        - word_mask (torch.Tensor): Mask indicating padding tokens of shape (batch_size, max_length).

        Returns:
        - List of lists containing community assignments for each node in each graph in the batch.
        - Tensor of padded incidence matrices for each graph in the batch, structured as [batch_size, max_length, max_number of communities].
        """
        self.embeddings = embeddings
        self.threshold = self.args.kernighan_lin_threshold
        batch_size, num_nodes, embedding_dim = embeddings.size()
        
        communities_batch = []
        incidence_matrices_batch = []
        max_num_hyperedges = 0
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust rows and cols based on batch size
        axes = axes.flatten()

        for batch_idx in range(batch_size):
            tokens = tokens_batch[batch_idx]
    
            # Get non-padding token indices
            non_padding_indices = torch.nonzero(word_mask[batch_idx]).squeeze()
            tokens = [tokens[i.item()] for i in non_padding_indices]
            num_tokens = len(tokens)
            
            # Extract embeddings and adjacency matrix for non-padding tokens
            embeddings_batch = embeddings[batch_idx, non_padding_indices]
            
            adj_matrix = torch.mm(embeddings_batch, embeddings_batch.transpose(0, 1))
            
            # Apply thresholding to convert to binary adjacency matrix
            adj_matrix_binary = (adj_matrix > self.threshold).float()
            
            # Create padded adjacency matrix
            padded_adj_matrix = torch.full((self.args.max_length, self.args.max_length), -1.0)
            padded_adj_matrix[non_padding_indices[:, None], non_padding_indices] = adj_matrix_binary
            
            # Convert adjacency matrix to networkx graph
            G = nx.Graph()
            max_length = padded_adj_matrix.shape[0]
            
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if adj_matrix_binary[i, j] > 0:
                        G.add_edge(i, j)
            
            print(f"Graph for {text_list[batch_idx]}:")
            print("Number of nodes:", G.number_of_nodes())
            print("Number of edges:", G.number_of_edges())
        
            # Perform Kernighan-Lin community detection
            communities = nx.algorithms.community.kernighan_lin_bisection(G)

            # Convert communities to a list of sets
            hyperedges = [set(communities[0]), set(communities[1])]

            # Determine number of nodes and hyperedges
            num_nodes = max_length
            num_hyperedges = len(hyperedges)
            
            # Update max_num_hyperedges if current graph has more hyperedges
            if num_hyperedges > max_num_hyperedges:
                max_num_hyperedges = num_hyperedges

            # Initialize incidence matrix
            incidence_matrix = np.full((num_nodes, num_hyperedges), -1)

            # Populate incidence matrix based on hyperedges
            for h_idx, hyperedge in enumerate(hyperedges):
                for node_idx in hyperedge:
                    incidence_matrix[node_idx, h_idx] = 1

            # Append results to batch lists
            communities_batch.append(communities)
            incidence_matrices_batch.append(incidence_matrix)
            
            # Visualize the graph with communities
            ax = axes[batch_idx]
            pos = nx.spring_layout(G)
            
            # Assign a unique color to each community
            colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
            for community in communities:
                color = next(colors)
                nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=500, alpha=0.8, ax=ax)
            
            nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
            nx.draw_networkx_labels(G, pos, labels={i: tokens[i] for i in range(num_tokens)}, font_size=10, ax=ax)
            
            ax.set_title(f"Graph for {text_list[batch_idx]}")
        
        plt.tight_layout()
        plt.show()

        # Pad incidence matrices to the max number of hyperedges and convert to tensor
        padded_incidence_matrices_batch = []
        for incidence_matrix in incidence_matrices_batch:
            padded_incidence_matrix = np.pad(incidence_matrix, ((0, 0), (0, max_num_hyperedges - incidence_matrix.shape[1])), mode='constant', constant_values=-1)
            padded_incidence_matrices_batch.append(torch.tensor(padded_incidence_matrix, dtype=torch.float32))

        # Stack all incidence matrices to get the final tensor of shape [batch_size, max_length, max_num_hyperedges]
        padded_incidence_matrices_batch = torch.stack(padded_incidence_matrices_batch)

        return communities_batch, padded_incidence_matrices_batch


    def community_to_incidence(self, num_nodes, partition):
        """
        Convert community assignments to an incidence matrix.

        Args:
        - num_nodes (int): Number of nodes in the graph.
        - partition (list of sets): List of sets, where each set contains nodes belonging to a community.

        Returns:
        - numpy array: Incidence matrix where rows represent nodes and columns represent communities.
        """
        num_communities = len(partition)
        incidence_matrix = np.zeros((num_nodes, num_communities), dtype=int)

        for comm_idx, community in enumerate(partition):
            for node in community:
                incidence_matrix[node, comm_idx] = 1

        return incidence_matrix
        
    
    
    
    
    
    




# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from transformers import BertTokenizer, BertModel

# class DependencyGraphParser:
#     def __init__(self, bert_model_name='bert-base-uncased'):
#         self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#         self.bert_model = BertModel.from_pretrained(bert_model_name)
#         self.pos_to_index = self._create_pos_mapping()
    
#     def _create_pos_mapping(self):
#         # Create a mapping for POS tags to indices
#         pos_tags = [
#             'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 
#             'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 
#             'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 
#             '-LRB-', '-RRB-', '.', ',', '``', "''", ':', '$', '#'
#         ]
#         return {tag: idx for idx, tag in enumerate(pos_tags)}
    
#     def _get_bert_embeddings(self, tokens):
#         inputs = self.tokenizer(tokens, return_tensors='pt', is_split_into_words=True, padding=True, truncation=True)
#         outputs = self.bert_model(**inputs)
#         return outputs.last_hidden_state.squeeze(0)
    
#     def _get_pos_embeddings(self, pos_tags):
#         pos_indices = [self.pos_to_index[tag] for tag in pos_tags]
#         return torch.tensor(pos_indices, dtype=torch.long).unsqueeze(1)
    
# def prepare_graph_data(data_point, aspect_post):
#     tokens = data_point['text_list']
#     pos_tags = data_point['pos']
#     head = data_point['head']
#     deprel = data_point['deprel']
#     aspect_range = range(aspect_post[0], aspect_post[1] + 1)
    
#     # Dummy token embeddings (you should replace this with actual embeddings)
#     token_embeddings = torch.eye(len(tokens))
    
#     # POS tag embeddings (dummy one-hot encoding, replace with actual embeddings)
#     pos_to_index = {pos: i for i, pos in enumerate(set(pos_tags))}
#     pos_embeddings = torch.eye(len(pos_to_index))
#     pos_features = torch.stack([pos_embeddings[pos_to_index[tag]] for tag in pos_tags])
    
#     # Aspect feature (binary feature indicating if token is part of the aspect term)
#     aspect_feature = torch.zeros(len(tokens), 1)
#     for i in aspect_range:
#         aspect_feature[i] = 1.0
    
#     # Combine features
#     node_features = torch.cat([token_embeddings, pos_features, aspect_feature], dim=1)
    
#     # Create edge index and edge attributes
#     edge_index = []
#     for i, head_idx in enumerate(head):
#         if head_idx > 0:  # Exclude ROOT
#             edge_index.append([head_idx - 1, i])
#             edge_index.append([i, head_idx - 1])
#     edge_index = torch.tensor(edge_index).t().contiguous()
    
#     data = Data(x=node_features, edge_index=edge_index)
#     return data

# # Example usage
# data_point = {
#     'text': 'i had of course bought a 3 year warranty , so i sent it in to be replaced and ( almost 2 months later ) the dv4 is what the sent me as a replacement .',
#     'aspect': '3 year warranty',
#     'pos': ['PRP', 'VBD', 'IN', 'NN', 'VBD', 'DT', 'CD', 'NN', 'NN', ',', 'IN', 'PRP', 'VBD', 'PRP', 'IN', 'TO', 'VB', 'VBN', 'CC', '-LRB-', 'RB', 'CD', 'NNS', 'RB', '-RRB-', 'DT', 'NN', 'VBZ', 'WP', 'DT', 'VBN', 'PRP', 'IN', 'DT', 'NN', '.'],
#     'post': [-6, -5, -4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#     'head': [2, 0, 4, 2, 2, 9, 8, 9, 5, 5, 5, 13, 5, 13, 13, 18, 18, 13, 18, 24, 22, 23, 24, 28, 24, 27, 28, 18, 28, 29, 30, 31, 35, 35, 31, 2],
#     'deprel': ['nsubj', 'ROOT', 'case', 'nmod', 'dep', 'det', 'compound', 'amod', 'dobj', 'punct', 'dep', 'nsubj', 'parataxis', 'dobj', 'compound:prt', 'mark', 'auxpass', 'dep', 'cc', 'punct', 'advmod', 'nummod', 'nmod:npmod', 'dep', 'punct', 'det', 'nsubj', 'conj', 'nsubj', 'dep', 'acl', 'dobj', 'case', 'det', 'nmod', 'punct'],
#     'length': 36,
#     'label': 'neutral',
#     'mask': [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'aspect_post': [6, 9],
#     'text_list': ['I', 'had', 'of', 'course', 'bought', 'a', '3', 'year', 'warranty', ',', 'so', 'I', 'sent', 'it', 'in', 'to', 'be', 'replaced', 'and', '(', 'almost', '2', 'months', 'later', ')', 'the', 'dv4', 'is', 'what', 'the', 'sent', 'me', 'as', 'a', 'replacement', '.']
# }

# parser = DependencyGraphParser()
# graph_data = parser.prepare_graph_data(data_point)
# print(graph_data)

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

# class CustomGNN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
#         super(CustomGNN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, output_dim)
#         self.fc = torch.nn.Linear(output_dim, num_classes)
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
        
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.conv3(x, edge_index)
        
#         # Compute adjacency matrix
#         adj_matrix = torch.mm(x, x.t())
        
#         # Compute logits for node classification
#         logits = self.fc(x)
        
#         return adj_matrix, logits

# # Example usage
# parser = DependencyGraphParser()
# data_point = {
#     'text': 'i had of course bought a 3 year warranty , so i sent it in to be replaced and ( almost 2 months later ) the dv4 is what the sent me as a replacement .',
#     'aspect': '3 year warranty',
#     'pos': ['PRP', 'VBD', 'IN', 'NN', 'VBD', 'DT', 'CD', 'NN', 'NN', ',', 'IN', 'PRP', 'VBD', 'PRP', 'IN', 'TO', 'VB', 'VBN', 'CC', '-LRB-', 'RB', 'CD', 'NNS', 'RB', '-RRB-', 'DT', 'NN', 'VBZ', 'WP', 'DT', 'VBN', 'PRP', 'IN', 'DT', 'NN', '.'],
#     'post': [-6, -5, -4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#     'head': [2, 0, 4, 2, 2, 9, 8, 9, 5, 5, 5, 13, 5, 13, 13, 18, 18, 13, 18, 24, 22, 23, 24, 28, 24, 27, 28, 18, 28, 29, 30, 31, 35, 35, 31, 2],
#     'deprel': ['nsubj', 'ROOT', 'case', 'nmod', 'dep', 'det', 'compound', 'amod', 'dobj', 'punct', 'dep', 'nsubj', 'parataxis', 'dobj', 'compound:prt', 'mark', 'auxpass', 'dep', 'cc', 'punct', 'advmod', 'nummod', 'nmod:npmod', 'dep', 'punct', 'det', 'nsubj', 'conj', 'nsubj', 'dep', 'acl', 'dobj', 'case', 'det', 'nmod', 'punct'],
#     'length': 36,
#     'label': 'neutral',
#     'mask': [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'aspect_post': [6, 9],
#     'text_list': ['I', 'had', 'of', 'course', 'bought', 'a', '3', 'year', 'warranty', ',', 'so', 'I', 'sent', 'it', 'in', 'to', 'be', 'replaced', 'and', '(', 'almost', '2', 'months', 'later', ')', 'the', 'dv4', 'is', 'what', 'the', 'sent', 'me', 'as', 'a', 'replacement', '.']
# }

# graph_data = parser.prepare_graph_data(data_point)

# input_dim = graph_data.x.shape[1]
# hidden_dim = 64
# output_dim = 32
# num_classes = len(parser.pos_to_index)  # Assuming number of classes equals number of POS tags

# model = CustomGNN(input_dim, hidden_dim, output_dim, num_classes)
# adj_matrix, logits = model(graph_data)

# print("Adjacency Matrix:")
# print(adj_matrix)

# print("Logits:")
# print(logits)

