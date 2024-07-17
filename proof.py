import numpy as np
import torch
import math
import scipy

class Proof1(object):
    def __init__(self, hypergraph, args, config):
        self.hypergraph = hypergraph
        self.seq_len = args.max_length
        self.k = config.top_k_knn
        
    def calculate_information_content(self, hypergraph):
        """
        Calculate the information content of a hypergraph.
        """
        edge_counts = np.sum(hypergraph, axis=0)
        total_edges = np.sum(edge_counts)
        edge_probabilities = edge_counts / total_edges
        entropy = -np.sum(edge_probabilities * np.log2(edge_probabilities + 1e-10))
        information_content = entropy  # Changed to positive entropy
        return information_content
    
    def generate_random_hypergraph(self, num_vertices, num_edges):
        """
        Generate a random hypergraph with the same dimensions as the constructed one.
        """
        return np.random.randint(0, 2, size=(num_vertices, num_edges))
        
    def compute_p_e(self):
        """
        Compute the probability distribution of hyperedges in the hypergraph.
        """
        batch_size, seq_len, _, _ = self.hypergraph.shape
        total_edges = batch_size * seq_len
        
        # Flatten the hypergraph
        flat_hypergraph = self.hypergraph.reshape(-1, seq_len)
        
        # Count unique hyperedges
        _, counts = np.unique(flat_hypergraph, axis=0, return_counts=True)
        
        # Compute p(e)
        p_e = counts.astype(float) / total_edges
        
        return p_e
    
    def compute_I_Hi(self):
        p_e = self.compute_p_e()
        I_Hi = -np.sum(p_e * np.log2(p_e + 1e-10)) # Added small number to prevent log(0) in denom
        return I_Hi
    
    def compute_I_Hr(self):
        seq_len = self.seq_len
        k = self.k
        p_e_r = 1 / scipy.special.comb(seq_len, k)
        I_Hr = -np.log2(p_e_r) 
        return I_Hr

def verify_lemma_1(model, epoch, args, config, proof=None):
    """
    Verify Lemma 1 for the current hypergraph.
    
    Args:
    model (HGSCANWrapper): The model containing the hypergraph
    epoch (int): Current epoch number
    args (object): Arguments object containing necessary parameters
    config (object): Config object containing necessary parameters
    proof (Proof1, optional): Existing Proof1 object, if any
    
    Returns:
    Proof1: The Proof1 object (either existing or newly created)
    bool: Whether Lemma 1 holds for this hypergraph
    """
    hypergraph = model.hypergraph
    
    if hypergraph is not None:
        if proof is None:
            # Initialize Proof1 object with the first valid hypergraph
            proof = Proof1(hypergraph.detach().cpu().numpy(), args, config)
        
        # Calculate I(H_i)
        I_Hi = proof.compute_I_Hi()
        
        # Calculate I(H_r)
        I_Hr = proof.compute_I_Hr()
        
        # Log the information content
        print(f"Epoch {epoch}, I(H_i): {I_Hi}, I(H_r): {I_Hr}")
        
        # Verify Lemma 1
        lemma_holds = I_Hi > I_Hr
        if lemma_holds:
            print("Lemma 1 holds: I(H_i) > I(H_r)")
        else:
            print("Warning: Lemma 1 does not hold for this batch")
        
        return proof, lemma_holds
    
    return proof, None