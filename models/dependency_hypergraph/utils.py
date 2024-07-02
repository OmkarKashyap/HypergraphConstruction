import torch
from torch import nn, optim
from torch.optim import Optimizer
import nltk
import spacy
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spacy model for dependency parsing
nlp = spacy.load('en_core_web_sm')

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
PAD_ID = -1
UNK_ID = 1

SUBJ_NER_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}
OBJ_NER_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}
NER_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}
POS_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

INFINITY_NUMBER = 1e12


def get_positions(start_idx, end_idx, length):
    """Get subj/obj position sequence."""
    if start_idx is None or end_idx is None:
        # If indices are not found, return a neutral sequence
        return [0] * length
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))

def get_subj_obj_indices(sentence):
    """
    Extract subject and object indices from a sentence using spacy for dependency parsing.
    Handles compound subjects and objects and provides more accurate start and end indices.
    """
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Parse sentence using spacy
    doc = nlp(sentence)
    
    # Initialize variables to hold subject and object start/end indices
    subj_start, subj_end = None, None
    obj_start, obj_end = None, None

    def find_full_span(token):
        """Find the full span of a multi-word token (e.g., compound subjects/objects)."""
        start, end = token.i, token.i
        for child in token.children:
            if child.dep_ in ['compound', 'amod', 'det']:
                start = min(start, child.i)
                end = max(end, child.i)
        return start, end
    
    # Use dictionaries to collect subject and object indices
    subj_indices = defaultdict(list)
    obj_indices = defaultdict(list)

    # Identify the subject and object tokens
    for token in doc:
        if 'subj' in token.dep_:
            start, end = find_full_span(token)
            subj_indices[start].append(end)
        if 'obj' in token.dep_:
            start, end = find_full_span(token)
            obj_indices[start].append(end)
    
    # Find the earliest starting point and the latest ending point for subjects and objects
    if subj_indices:
        subj_start = min(subj_indices.keys())
        subj_end = max([max(ends) for ends in subj_indices.values()])
    if obj_indices:
        obj_start = min(obj_indices.keys())
        obj_end = max([max(ends) for ends in obj_indices.values()])
    
    return subj_start, subj_end, obj_start, obj_end

def get_subj_obj_positions(sentences):
    """
    Get subject and object positions for a list of sentences.
    
    Parameters:
    sentences (list of str): List of sentences.

    Returns:
    list of dict: List of dictionaries containing 'subj_pos' and 'obj_pos' for each sentence.
    """
    positions_list = []
    for sentence in sentences:
        subj_start, subj_end, obj_start, obj_end = get_subj_obj_indices(sentence)
        length = len(sentence.split())
        
        subj_positions = get_positions(subj_start, subj_end, length)
        obj_positions = get_positions(obj_start, obj_end, length)
        
        positions_list.append({'subj_pos': subj_positions, 'obj_pos': obj_positions})
    
    return positions_list

def generate_subj_pos_tensor(texts):
    subj_pos_list = []
    
    # Process each text
    for text in texts:
        # Tokenize the text using spaCy
        doc = nlp(text)
        
        # Initialize a list to store subject positions for this text
        subj_positions = []
        
        # Iterate through sentences in the document
        for sent in doc.sents:
            # Extract subjects from the sentence
            subjects = [token.i - sent.start for token in sent if token.dep_ == "nsubj"]
            
            # Append subjects' positions to the list
            subj_positions.extend(subjects)
        
        # Sort and convert positions to a tensor
        subj_positions.sort()
        subj_positions_tensor = torch.tensor(subj_positions, dtype=torch.long)
        
        # Append the tensor to the list
        subj_pos_list.append(subj_positions_tensor)
    
    # Pad tensors to ensure they have the same length (if needed)
    max_len = max(len(tensor) for tensor in subj_pos_list)
    subj_pos_tensors_padded = []
    for tensor in subj_pos_list:
        pad_length = max_len - len(tensor)
        padded_tensor = torch.cat([tensor, torch.zeros(pad_length, dtype=torch.long)])
        subj_pos_tensors_padded.append(padded_tensor)
    
    # Stack tensors into a single batch tensor
    subj_pos_batch_tensor = torch.stack(subj_pos_tensors_padded)
    
    return subj_pos_batch_tensor

def generate_obj_pos_tensor(texts):
    obj_pos_list = []
    
    # Process each text
    for text in texts:
        # Tokenize the text using spaCy
        doc = nlp(text)
        
        # Initialize a list to store object positions for this text
        obj_positions = []
        
        # Iterate through sentences in the document
        for sent in doc.sents:
            # Extract objects from the sentence
            objects = [token.i - sent.start for token in sent if token.dep_ == "dobj" or token.dep_ == "obj"]
            
            # Append objects' positions to the list
            obj_positions.extend(objects)
        
        # Sort and convert positions to a tensor
        obj_positions.sort()
        obj_positions_tensor = torch.tensor(obj_positions, dtype=torch.long)
        
        # Append the tensor to the list
        obj_pos_list.append(obj_positions_tensor)
    
    # Pad tensors to ensure they have the same length (if needed)
    max_len = max(len(tensor) for tensor in obj_pos_list)
    obj_pos_tensors_padded = []
    for tensor in obj_pos_list:
        pad_length = max_len - len(tensor)
        padded_tensor = torch.cat([tensor, torch.zeros(pad_length, dtype=torch.long)])
        obj_pos_tensors_padded.append(padded_tensor)
    
    # Stack tensors into a single batch tensor
    obj_pos_batch_tensor = torch.stack(obj_pos_tensors_padded)
    
    return obj_pos_batch_tensor
# Function to get NER tags for the entire text
def get_ner(text):
    doc = nlp(text)
    ner_tags = [NER_TO_ID.get(ent.label_, UNK_TOKEN) for ent in doc.ents]
    return ner_tags

# Function to identify the subject in a sentence
def identify_subject(doc):
    for token in doc:
        if token.dep_ in {'nsubj', 'nsubjpass'}:
            return token
    return None

# Function to identify the object in a sentence
def identify_object(doc):
    for token in doc:
        if token.dep_ in {'dobj', 'pobj', 'attr', 'dative'}:
            return token
    return None

# Function to get POS tag for the subject
def get_subj_pos(text):
    doc = nlp(text)
    subject = identify_subject(doc)
    if subject:
        return POS_TO_ID.get(subject.tag_, UNK_TOKEN)
    return POS_TO_ID[UNK_TOKEN]

# Function to get POS tag for the object
def get_obj_pos(text):
    doc = nlp(text)
    obj = identify_object(doc)
    if obj:
        return POS_TO_ID.get(obj.tag_, UNK_TOKEN)
    return POS_TO_ID[UNK_TOKEN]

# Function to get NER type for the subject
def get_subj_type(text):
    doc = nlp(text)
    subject = identify_subject(doc)
    if subject:
        for ent in doc.ents:
            if ent.start <= subject.i < ent.end:
                return SUBJ_NER_TO_ID.get(ent.label_, UNK_TOKEN)
    return SUBJ_NER_TO_ID[UNK_TOKEN]

# Function to get NER type for the object
def get_obj_type(text):
    doc = nlp(text)
    obj = identify_object(doc)
    if obj:
        for ent in doc.ents:
            if ent.start <= obj.i < ent.end:
                return OBJ_NER_TO_ID.get(ent.label_, UNK_TOKEN)
    return OBJ_NER_TO_ID[UNK_TOKEN]

### class
class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation 
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) *\
                        init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss

### torch specific functions
def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name in ['adagrad', 'myadagrad']:
        # use my own adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2) # use default lr
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
    
def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        

if __name__ == "__main__":
# Example usage
    text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."
    print("NER tags:", get_ner(text))
    print("Subject POS tag:", get_subj_pos(text))
    print("Object POS tag:", get_obj_pos(text))
    print("Subject type:", get_subj_type(text))
    print("Object type:", get_obj_type(text))
