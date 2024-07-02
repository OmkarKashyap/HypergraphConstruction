import torch 
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

from models.dependency_hypergraph.utils import get_optimizer, change_lr, get_subj_obj_positions
from models.dependency_hypergraph.tree import inputs_to_tree_reps

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
PAD_ID = -1
UNK_ID = 1

SUBJ_NER_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}
OBJ_NER_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}
NER_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}
POS_TO_ID = {PAD_TOKEN: -1, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

INFINITY_NUMBER = 1e12

class GCNRelationalModel(nn.Module):
    def __init__(self, args, embedding_matrix=None) -> None:
        super().__init__()
        
        self.args=args
        self.embedding_matrix=embedding_matrix
        
        self.emb = nn.Embedding(3860, args.dim_in) ######### 
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float).clone().detach(), freeze=True)
        self.pos_emb = nn.Embedding(len(POS_TO_ID), args.pos_dim)  
        
        embeddings = self.emb
        # embeddings = (self.emb, self.pos_emb)
        # self.init_embeddings()
        
        self.gcn = GCN(args, embeddings)
        layers = [nn.Linear(self.args.dim_in, self.args.hidden_dim), nn.ReLU()]
        for _ in range(self.args.mlp_layers-1):
            layers += [nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU()]
        # self.out_mlp = nn.Sequential(*layers)
        self.out_mlp = nn.ModuleList([
            nn.Linear(900, self.args.hidden_dim),
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
            # Add more linear layers as needed...
        ])
        
    def forward(self, inputs, plain_text):
        (feats,tokens, aspect,post,pos,deprel,adj,mask,head) = inputs
        words = tokens
        subj_obj_positions = get_subj_obj_positions(plain_text)
        subj_pos = [torch.tensor(pos['subj_pos']) for pos in subj_obj_positions]
        obj_pos = [torch.tensor(pos['obj_pos']) for pos in subj_obj_positions] 
        
        subj_pos = pad_sequence(subj_pos, batch_first=True, padding_value=0)
        obj_pos = pad_sequence(obj_pos, batch_first=True, padding_value=0)
        
        data_len = ([len(tensor[tensor == 0]) for tensor in mask])
        maxlen = max(data_len)
        self.args.maxlen = maxlen
        
        head = pad_sequence(head, batch_first=True, padding_value=-1)
        words = pad_sequence(words, batch_first=True, padding_value=-1)
        adj = inputs_to_tree_reps(head, words, data_len, self.args.prune_k, subj_pos, obj_pos, maxlen)
        h, pool_mask = self.gcn(adj, inputs)
        
        # pooling
        subj_mask = subj_pos.eq(0).unsqueeze(2)[:, :self.args.maxlen, :]
        obj_mask = obj_pos.eq(0).unsqueeze(2)[:, :self.args.maxlen, :]
        pool_type = self.args.pooling
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        for layer in self.out_mlp:
            outputs = layer(outputs)
        return outputs, h_out          
        
    def init_embeddings(self, agg_fn = False):
        if self.embedding_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0,-1.0)
        else:
            # self.embedding_matrix = torch.from_numpy(self.embedding_matrix)
            self.emb.weight.data.copy_(self.embedding_matrix)
            
        # try to concat them if some variable says that
        if agg_fn==True:
            if agg_fn == 'concat':
                embeddings =  (self.emb, self.pos_emb)
            if agg_fn == 'sum':
                embeddings = (self.emb + self.pos_emb )
            if agg_fn =='mean':
                embeddings = ((self.emb + self.pos_emb)/2)
        

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, args, embeddings):
        super(GCN, self).__init__()
        self.args = args
        # self.layers = num_layers
        # self.mem_dim = mem_dim
        self.in_dim = args.dim_in +args.pos_dim 
        self.emb = embeddings
        # self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if getattr(self.args, 'rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, args.rnn_hidden, args.rnn_layers, batch_first=True, \
                    dropout=args.rnn_dropout, bidirectional=True)
            self.in_dim = args.rnn_hidden * 2
            self.rnn_drop = nn.Dropout(args.rnn_dropout) # use on last layer output

        self.in_drop = nn.Dropout(args.dropout_rate)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.args.n_layers):
            input_dim = self.in_dim if layer == 0 else self.args.rnn_mem_dim
            self.W.append(nn.Linear(input_dim, self.args.rnn_mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        PAD_ID = -1  # Assuming PAD_ID is defined as -1
        seq_lens = [tensor.eq(PAD_ID).logical_not().sum().item() for tensor in masks]
        seq_lens_sorted, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
        rnn_inputs_sorted = rnn_inputs[sorted_indices]
    
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, self.args.rnn_layers)
        
        # Adjust seq_lens_sorted to ensure it aligns with desired sequence length
        max_length = self.args.maxlen  # Desired maximum sequence length
        seq_lens_sorted = torch.clamp(seq_lens_sorted, max=max_length)  # Limit sequence lengths to max_length
        
        rnn_inputs_packed = rnn_utils.pack_padded_sequence(rnn_inputs_sorted, seq_lens_sorted.tolist(), batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs_packed, (h0, c0))
        rnn_outputs, _ = rnn_utils.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=max_length)
    
        # Optionally, reorder rnn_outputs to match the original order
        _, original_order = sorted_indices.sort()
        rnn_outputs = rnn_outputs[original_order]
    
        return rnn_outputs

    def forward(self, adj, inputs):
        x, words, aspect,post, pos, deprel, adj2,mask,head = inputs # unpack
        words = torch.stack(words)
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if getattr(self.args, 'rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, mask, words.size()[0]))
        else:
            gcn_inputs = embs
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if hasattr(self.args, 'no_adj') and self.args.no_adj:
            adj = torch.zeros_like(adj)

        for l in range(self.args.n_layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.args.n_layers - 1 else gAxW

        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    # if use_cuda:
    #     return h0.cuda(), c0.cuda()
    # else:
    return h0, c0
    

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationalModel(args, embedding_matrix=emb_matrix)
        in_dim = args.hidden_dim
        self.classifier = nn.Linear(in_dim, args.n_categories)
        self.args = args

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs, plain_text):
        outputs, pooling_output = self.gcn_model(inputs, plain_text)
        logits = self.classifier(outputs)
        return logits, pooling_output
    
class Trainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, args, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch['text'][:args.batch_size]]
        labels = Variable(batch['polarity'].cuda())
    else:
        inputs = [
            Variable(torch.stack([
                batch['text'][i], 
                batch['aspect'][i], 
                batch['post'][i], 
                batch['pos'][i], 
                batch['deprel'][i], 
                batch['mask'][i], 
                batch['adj'][i], 
                batch['head'][i]
            ]))
            for i in range(min(args.batch_size, len(batch['text'])))
        ]
        labels = [
            Variable(batch['polarity'][i])
            for i in range(min(args.batch_size, len(batch['text'])))
        ]
    
    feats = batch['text']
    aspect = batch['aspect']
    post = batch['post']
    pos = batch['pos']
    deprel = batch['deprel']
    adj = batch['adj']
    mask = batch['mask']
    polarity = batch['polarity']
    plain_text = batch['plain_text']
    head = batch['head']
    
    return inputs, labels, feats, aspect, post, pos, deprel, adj, mask, polarity, plain_text, head

        
class GCNTrainer(Trainer):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(args, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if args.cuda:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = get_optimizer(args.optim, self.parameters, args.learning_rate)

    def update(self, batch):
        inputs, labels, feats, aspect, post, pos, deprel,adj,mask,polarity, plain_text,head = unpack_batch(batch, self.args, self.args.cuda)

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.args.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.args.conv_l2
        # l2 penalty on output representations
        if self.args.get('pooling_l2', 0) > 0:
            loss += self.args.pooling_l2 * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, feats, aspect, post, pos, deprel,adj,mask,polarity, plain_text,head = unpack_batch(batch, self.args, self.args.cuda)
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        # if unsort:
        #     _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
        #             predictions, probs)))]
        return predictions, probs, loss.item()
        
        
# class DependencyHypergraph(nn.Module):
#     def __init__(self, args) -> None:
#         super().__init__()
        
#     def forward(self, x):
#         dependency_graph  = create_dependency_graph(x)
#         graph_tensor = graph_to_torch_data(dependency_graph)
        
#         model = GCN()
#         out = model(graph_tensor)
        
#         community_detection = CommunityDetection()
#         communities = community_detection.Louvain(dependency_graph, node_embeddings=out)
#         hypergraph = create_hypergraph(communities)
#         return hypergraph