# import os
# import sys
# import re
# import json
# import pickle
# import numpy as np
# from tqdm import tqdm
# from transformers import BertTokenizer
# from torch.utils.data import Dataset
# from models.layers import HGConstruct

# import torch
# import torch.nn as nn 


# def ParseData(data_path):
#     with open(data_path) as infile:
#         all_data = []
#         data = json.load(infile)
#         for d in data:
#             for aspect in d['aspects']:
#                 text_list = list(d['token'])
#                 tok = list(d['token'])       # word token
#                 length = len(tok)            # real length
#                 # if args.lower == True:
#                 tok = [t.lower() for t in tok]
#                 tok = ' '.join(tok)
#                 asp = list(aspect['term'])   # aspect
#                 asp = [a.lower() for a in asp]
#                 asp = ' '.join(asp)
#                 label = aspect['polarity']   # label                
#                 pos = list(d['pos'])         # pos_tag 
#                 head = list(d['head'])       # head
#                 deprel = list(d['deprel'])   # deprel
#                 # position
#                 aspect_post = [aspect['from'], aspect['to']] 
#                 post = [i-aspect['from'] for i in range(aspect['from'])] \
#                        +[0 for _ in range(aspect['from'], aspect['to'])] \
#                        +[i-aspect['to']+1 for i in range(aspect['to'], length)]
#                 # aspect mask
#                 if len(asp) == 0:
#                     mask = [1 for _ in range(length)]    # for rest16
#                 else:
#                     mask = [0 for _ in range(aspect['from'])] \
#                        +[1 for _ in range(aspect['from'], aspect['to'])] \
#                        +[0 for _ in range(aspect['to'], length)]
                
#                 sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
#                           'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
#                           'aspect_post': aspect_post, 'text_list': text_list}
#                 all_data.append(sample)

#     return all_data


# def build_tokenizer(fnames, max_length, data_file):
#     parse = ParseData
#     if os.path.exists(data_file):
#         print('loading tokenizer:', data_file)
#         tokenizer = pickle.load(open(data_file, 'rb'))
#     else:
#         tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
#         pickle.dump(tokenizer, open(data_file, 'wb'))
#     return tokenizer


# class Vocab(object):
#     ''' vocabulary of dataset '''
#     def __init__(self, vocab_list, add_pad, add_unk):
#         self._vocab_dict = dict()
#         self._reverse_vocab_dict = dict()
#         self._length = 0
#         if add_pad:
#             self.pad_word = '<pad>'
#             # self.pad_id = self._length
#             self.pad_id = -1
#             self._length += 1
#             self._vocab_dict[self.pad_word] = self.pad_id
#         if add_unk:
#             self.unk_word = '<unk>'
#             self.unk_id = self._length
#             self._length += 1
#             self._vocab_dict[self.unk_word] = self.unk_id
#         for w in vocab_list:
#             self._vocab_dict[w] = self._length
#             self._length += 1
#         for w, i in self._vocab_dict.items():   
#             self._reverse_vocab_dict[i] = w  
    
#     def word_to_id(self, word):  
#         if hasattr(self, 'unk_id'):
#             return self._vocab_dict.get(word, self.unk_id)
#         return self._vocab_dict[word]
    
#     def id_to_word(self, id_):   
#         if hasattr(self, 'unk_word'):
#             return self._reverse_vocab_dict.get(id_, self.unk_word)
#         return self._reverse_vocab_dict[id_]
    
#     def has_word(self, word):
#         return word in self._vocab_dict
    
#     def __len__(self):
#         return self._length
    
#     @staticmethod
#     def load_vocab(vocab_path: str):
#         with open(vocab_path, "rb") as f:
#             return pickle.load(f)

#     def save_vocab(self, vocab_path):
#         with open(vocab_path, "wb") as f:
#             pickle.dump(self, f)


# class Tokenizer(object):
#     ''' transform text to indices '''
#     def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
#         self.vocab = vocab
#         self.max_length = max_length
#         self.lower = lower

#         self.pos_char_to_int = pos_char_to_int
#         self.pos_int_to_char = pos_int_to_char
    
#     @classmethod
#     def from_files(cls, fnames, max_length, parse, lower=True):
#         corpus = set()
#         pos_char_to_int, pos_int_to_char = {}, {}
#         for fname in fnames:
#             for obj in parse(fname):
#                 text_raw = obj['text']
#                 if lower:
#                     text_raw = text_raw.lower()
#                 corpus.update(Tokenizer.split_text(text_raw)) 
#         return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
#     @staticmethod
#     def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
#         x = (np.zeros(maxlen) + pad_id).astype(dtype)
#         if truncating == 'pre':
#             trunc = sequence[-maxlen:] 
#         else:
#             trunc = sequence[:maxlen]
#         trunc = np.asarray(trunc, dtype=dtype)
#         if padding == 'post':
#             x[:len(trunc)] = trunc 
#         else:
#             x[-len(trunc):] = trunc
#         return x
    
#     def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
#         if self.lower:
#             text = text.lower()
#         words = Tokenizer.split_text(text)
#         sequence = [self.vocab.word_to_id(w) for w in words] 
#         if len(sequence) == 0:
#             sequence = [0]
#         if reverse:
#             sequence.reverse() 
            
#         return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
#                                       padding=padding, truncating=truncating)
    
#     @staticmethod
#     def split_text(text):
#         # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
#         #     text = text.replace(ch, " "+ch+" ")
#         return text.strip().split()

# class SentenceDataset(Dataset):
#     ''' PyTorch standard dataset class '''
#     def __init__(self, fname, tokenizer, args, vocab_help, embedding_matrix, save_path):

#         i = 0
#         inc_exists = False
#         if os.path.exists(save_path):
#             inc_exists = True
#             with open(save_path, 'rb') as f:
#                 self.incidences = pickle.load(f)
#         else:
#             self.incidences = []
        
#         self.construct = HGConstruct(args.eps, args.min_samples, args)
#         self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
#         parse = ParseData
#         post_vocab, pos_vocab, dep_vocab, pol_vocab, head_vocab = vocab_help
#         data = list()
#         polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
#         for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
                        
#             plain_text = obj['text']
#             text = tokenizer.text_to_sequence(obj['text'])


#             text_temp = text.copy()
#             text_temp = [0 if x == -1 else x for x in text_temp]
#             text_tensor = torch.as_tensor(text_temp, dtype=torch.int64)
            
#             x = self.embedding(text_tensor)
#             if not inc_exists:
#                 incidence_matrix = self.construct.cluster(x)
#                 self.incidences.append(incidence_matrix)
#                 i += 1
#             text = tokenizer.pad_sequence(text, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')

#             aspect = tokenizer.text_to_sequence(obj['aspect'])  
#             aspect = tokenizer.pad_sequence(aspect, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
#             pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
#             pos = tokenizer.pad_sequence(pos, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
#             post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
#             post = tokenizer.pad_sequence(post, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
#             head = tokenizer.pad_sequence(obj['head'], pad_id = args.pad_id, maxlen = args.max_length, dtype='int64', padding='post', truncating='post')
            
#             deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
#             deprel = tokenizer.pad_sequence(deprel, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
#             sentence_length = obj['length']
            
#             polarity = polarity_dict[obj['label']]
            
#             position_mask = tokenizer.pad_sequence(obj['mask'], pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
#             word_mask = [1 if i < len(obj['mask']) else 0 for i in range(args.max_length)]
            
#             aspect_post_start = obj['aspect_post'][0]
#             aspect_post_end = obj['aspect_post'][1]
            
#             adj = np.ones(args.max_length) * args.pad_id
            
#             text_list = obj['text_list']
            
#             data.append({
#                 'text': text, 
#                 'aspect': aspect, 
#                 'pos': pos,
#                 'post': post,
#                 'head':head,
#                 'deprel': deprel,
#                 'sentence_length': sentence_length,
#                 'polarity': polarity,
#                 'adj': adj,
#                 'pos_mask': position_mask,
#                 'word_mask': word_mask,
#                 'aspect_post_start': aspect_post_start,
#                 'aspect_post_end':aspect_post_end,
#                 'plain_text': plain_text,
#                 'text_list': text_list,
#                 'incidence_matrix': incidence_matrix if not inc_exists else self.incidences[i],
#             })
            
#         if not inc_exists:
#             with open(save_path, 'wb') as f:
#                 pickle.dump(self.incidences, f)
#         self._data = data

#     def __getitem__(self, index):
#         return self._data[index]
    
#     def __len__(self):
#         return len(self._data)


# def _load_wordvec(data_path, embed_dim, vocab=None):
#     with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
#         word_vec = dict()
#         if embed_dim == 200:
#             for line in f:
#                 tokens = line.rstrip().split()
#                 if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
#                     continue
#                 if vocab is None or vocab.has_word(tokens[0]):
#                     word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
#         elif embed_dim == 300:
#             for line in f:
#                 tokens = line.rstrip().split()
#                 if tokens[0] == '<pad>': # avoid them
#                     continue
#                 elif tokens[0] == '<unk>':
#                     word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
#                 word = ''.join((tokens[:-300]))
#                 if vocab is None or vocab.has_word(tokens[0]):
#                     word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
#         else:
#             print("embed_dim error!!!")
#             exit()
            
#         return word_vec

# def build_embedding_matrix(vocab, embed_dim, data_file):
#     if os.path.exists(data_file):
#         print('loading embedding matrix:', data_file)
#         embedding_matrix = pickle.load(open(data_file, 'rb'))
#     else:
#         print('loading word vectors...')
#         embedding_matrix = np.zeros((len(vocab), embed_dim))
#         fname = 'glove/glove.840B.300d.txt'
#         word_vec = _load_wordvec(fname, embed_dim, vocab)
#         for i in range(len(vocab)):
#             vec = word_vec.get(vocab.id_to_word(i))
#             if vec is not None:
#                 embedding_matrix[i] = vec
#         pickle.dump(embedding_matrix, open(data_file, 'wb'))
#     return embedding_matrix


# def softmax(x):
#     if len(x.shape) > 1:
#         # matrix
#         tmp = np.max(x, axis=1)
#         x -= tmp.reshape((x.shape[0], 1))
#         x = np.exp(x)
#         tmp = np.sum(x, axis=1)
#         x /= tmp.reshape((x.shape[0], 1))
#     else:
#         # vector
#         tmp = np.max(x)
#         x -= tmp
#         x = np.exp(x)
#         tmp = np.sum(x)
#         x /= tmp
#     return x


# class Tokenizer4BertGCN:
#     def __init__(self, max_seq_len, pretrained_bert_name):
#         self.max_seq_len = max_seq_len
#         self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
#         self.cls_token_id = self.tokenizer.cls_token_id
#         self.sep_token_id = self.tokenizer.sep_token_id
#     def tokenize(self, s):
#         return self.tokenizer.tokenize(s)
#     def convert_tokens_to_ids(self, tokens):
#         return self.tokenizer.convert_tokens_to_ids(tokens)
    
# class SqueezeEmbedding(nn.Module):
#     '''
#     Squeeze sequence embedding length to the longest one in the batch
#     '''
#     def __init__(self, batch_first=True):
#         super(SqueezeEmbedding, self).__init__()
#         self.batch_first = batch_first
    
#     def forward(self, x, x_len):
#         '''
#         sequence -> sort -> pad and pack -> unpack -> unsort
#         '''
#         '''sort'''
#         x_sort_idx = torch.sort(x_len, descending=True)[1].long()
#         x_unsort_idx = torch.sort(x_sort_idx)[1].long()
#         x_len = x_len[x_sort_idx]
#         x = x[x_sort_idx]
#         '''pack'''
#         x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
#         '''unpack'''
#         out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
#         if self.batch_first:
#             out = out[x_unsort_idx]
#         else:
#             out = out[:, x_unsort_idx]
#         return out
        

# class Seq2Feats(nn.Module):
#     def __init__(self, embedding_matrix, args):
#         super(Seq2Feats, self).__init__()
        
#         # Initialize embedding layer from pre-trained matrix
#         self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
#         self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=True, padding_idx=args.pad_id)
        
        
#     def forward(self, inputs):
#         # Assuming inputs is a dictionary with 'text' and 'mask' keys
#         text_list = inputs['text']  # List of tensors, each shape [batch_size, max_length]
#         word_mask = inputs['word_mask']  # Tensor of shape [batch_size, max_length], with 0s indicating padding
        
#         # Stack tensors in the list
#         text = torch.stack(text_list)  # Shape [batch_size, max_length]
#         mask = torch.stack(word_mask) 
        
#         #mask text to remove pad as -1
#         text_masked = text * mask
        
#         # Apply embedding based on mask
#         x = self.embedding(text_masked.long())  # Shape [batch_size, max_length, embedding_dim]
        
#         return x
        
    
#     def remove_trailing_zeros(self, tensor):
#             last_non_zero_idx = (tensor != 0).nonzero(as_tuple=False).max()
#             return len(tensor[:last_non_zero_idx + 1])
        
    
import os
import sys
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
from models.layers import HGConstruct


from nltk.corpus import stopwords
import string
import torch
import torch.nn as nn 
import math

def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label                
                pos = list(d['pos'])         # pos_tag 
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel
                # position
                aspect_post = [aspect['from'], aspect['to']] 
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            # self.pad_id = self._length
            self.pad_id = -1
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w  
    
    def word_to_id(self, word):  
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):   
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char
    
    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw)) 
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:] 
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc 
        else:
            x[-len(trunc):] = trunc
        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words] 
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse() 
            
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
                                      padding=padding, truncating=truncating)
    
    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()

class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, args, config, vocab_help, embedding_matrix, save_path):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        i = 0
        inc_exists = False
        if os.path.exists(save_path):
            inc_exists = True
            with open(save_path, 'rb') as f:
                self.incidences = pickle.load(f)
        else:
            self.incidences = []
        
        self.construct = HGConstruct(args, config)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)

        self.positional_encoding = PositionalEncoding(d_model = embedding_matrix.size(1), max_len = args.max_length)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim = embedding_matrix.shape[1], num_heads=5, batch_first=True )
        
        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab, head_vocab = vocab_help
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text_list = obj['text_list'] + [obj['aspect']]           
            plain_text = obj['text'] + obj['aspect']
            text = tokenizer.text_to_sequence(obj['text'])

            text_temp = text.copy()
            text_temp = [0 if x == -1 else x for x in text_temp]
        
            updated_token_ids = [0 if text_list[i] in self.stop_words else text_temp[i] for i in range(len(text_list))]
            updated_token_ids = tokenizer.pad_sequence(updated_token_ids, pad_id=0, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            text_tensor = torch.as_tensor(updated_token_ids, dtype=torch.int64)
            
            x = self.embedding(text_tensor)
            x = self.positional_encoding(x)

            aspect_text = tokenizer.text_to_sequence(obj['aspect'])  
            aspect_text = [0 if x==-1 else x for x in aspect_text]
            aspect_tensor = tokenizer.pad_sequence(aspect_text, pad_id=0, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            aspect_tensor = torch.as_tensor(aspect_tensor, dtype=torch.int64)
            
            aspect_embedding = self.embedding(aspect_tensor)
            aspect_embedding = self.positional_encoding(aspect_embedding)

            word_mask = [i < len(obj['mask']) for i in range(args.max_length)]
            attn_mask = torch.tensor(word_mask, dtype=torch.bool).unsqueeze(0).to(torch.float32)

            x=x.to(torch.float32)
            aspect_embedding = aspect_embedding.to(torch.float32)
            attn_mask = attn_mask.to(torch.bool)

            x, attn_output_weights = self.multi_head_attention(query=aspect_embedding, key=x, value=x, key_padding_mask =attn_mask)

            x_detached = x.detach().squeeze(0)
            if not inc_exists:
                incidence_matrix = self.construct.cluster(x_detached)
                self.incidences.append(incidence_matrix)
                i += 1
            text = tokenizer.pad_sequence(text, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')

            
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
            head = tokenizer.pad_sequence(obj['head'], pad_id = args.pad_id, maxlen = args.max_length, dtype='int64', padding='post', truncating='post')
            
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
            sentence_length = obj['length']
            
            polarity = polarity_dict[obj['label']]
            
            position_mask = tokenizer.pad_sequence(obj['mask'], pad_id=args.pad_id, maxlen=args.max_length, dtype='int64', padding='post', truncating='post')
            
            word_mask = [1 if i < len(obj['mask']) else 0 for i in range(args.max_length)]
            
            aspect_post_start = obj['aspect_post'][0]
            aspect_post_end = obj['aspect_post'][1]
            
            adj = np.ones(args.max_length) * args.pad_id
            
            
            data.append({
                'text': text, 
                'aspect': aspect_tensor, 
                'pos': pos,
                'post': post,
                'head':head,
                'deprel': deprel,
                'sentence_length': sentence_length,
                'polarity': polarity,
                'adj': adj,
                'pos_mask': position_mask,
                'word_mask': word_mask,
                'aspect_post_start': aspect_post_start,
                'aspect_post_end':aspect_post_end,
                'plain_text': plain_text,
                'text_list': text_list,
                'incidence_matrix': incidence_matrix if not inc_exists else self.incidences[i],
            })
            
        if not inc_exists:
            with open(save_path, 'wb') as f:
                pickle.dump(self.incidences, f)
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)


def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>': # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()
            
        return word_vec

def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = 'glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

    
        
# class Seq2Feats(nn.Module):
#     def __init__(self, embedding_matrix, args):
#         super(Seq2Feats, self).__init__()
        
#         # Initialize embedding layer from pre-trained matrix
#         self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float).to(device=args.device)
#         self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=True, padding_idx=args.pad_id)
        
#     def forward(self, inputs):
#         # Assuming inputs is a dictionary with 'text' and 'mask' keys
#         text_list = inputs['text']  # List of tensors, each shape [batch_size, max_length]
#         word_mask = inputs['word_mask']  # Tensor of shape [batch_size, max_length], with 0s indicating padding
        
#         # Stack tensors in the list
#         text = torch.stack(text_list)  # Shape [batch_size, max_length]
#         mask = torch.stack(word_mask)
        
#         # Mask text to remove padding as -1 (assuming mask is already 0/1)
#         text_masked = text * mask  # Ensure mask is of type long
        
#         # Apply embedding based on masked text
#         x = self.embedding(text_masked)  # Shape [batch_size, max_length, embedding_dim]
        
#         return x
    
#     def remove_trailing_zeros(self, tensor):
#         last_non_zero_idx = (tensor != 0).nonzero(as_tuple=False).max()
#         return len(tensor[:last_non_zero_idx + 1])

class Seq2Feats(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(Seq2Feats, self).__init__()
        self.args=args
        # Initialize embedding layer from pre-trained matrix
        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float).to(device=args.device)
        self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=True, padding_idx=args.pad_id)
        
    def forward(self, inputs):
        # Assuming inputs is a dictionary with 'text' and 'mask' keys
        text_list = inputs['text']  # List of tensors, each shape [batch_size, max_length]
        word_mask = inputs['word_mask']  # Tensor of shape [batch_size, max_length], with 0s indicating padding
        
        # Stack tensors in the list
        text = torch.stack(text_list).to(device=self.args.device)  # Shape [batch_size, max_length]
        mask = torch.stack(word_mask).to(device=self.args.device)
        
        # Mask text to remove padding as -1 (assuming mask is already 0/1)
        text_masked = text * mask  # Ensure mask is of type long
        
        # Apply embedding based on masked text
        x = self.embedding(text_masked)  # Shape [batch_size, max_length, embedding_dim]
        
        return x
    
    def remove_trailing_zeros(self, tensor):
        last_non_zero_idx = (tensor != 0).nonzero(as_tuple=False).max()
        return len(tensor[:last_non_zero_idx + 1])
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)