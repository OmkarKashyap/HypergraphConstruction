import os
import sys
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from models.layers import HGConstruct


from nltk.corpus import stopwords
import string
import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel


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

def pad_sequence(sequence, pad_id, maxlen, dtype=torch.int64, padding='post', truncating='post'):
    if len(sequence) > maxlen:
        if truncating == 'post':
            sequence = sequence[:maxlen]
        elif truncating == 'pre':
            sequence = sequence[-maxlen:]
    else:
        padding_length = maxlen - len(sequence)
        if padding == 'post':
            sequence = sequence + [pad_id] * padding_length
        elif padding == 'pre':
            sequence = [pad_id] * padding_length + sequence
    return torch.tensor(sequence, dtype=dtype)

# class AspectAwareBERTEmbedding(nn.Module):
#     def __init__(self, bert_model='bert-base-uncased', hidden_size=768, max_length=128):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(bert_model)
#         self.tokenizer = BertTokenizer.from_pretrained(bert_model)
#         self.hidden_size = hidden_size
#         self.max_length = max_length
#         self.pad_id=0
#         self.max_length=85
        
#         self.aspect_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
#         self.rel_pos_embedding = nn.Embedding(2 * max_length - 1, hidden_size)
        
#     def forward(self, text, aspect, aspect_start, aspect_end, pos_mask):
        
#         inputs = self.tokenizer(text, aspect, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
#         inputs = pad_sequence(inputs, self.pad_id, self.max_length)

#         outputs = self.bert(**inputs)
#         sequence_output = outputs.last_hidden_state
#         # print(sequence_output.shape)
        
#         seq_length = sequence_output.size(1)
        
#         aspect_pos = (aspect_start + aspect_end) // 2  # Middle of the aspect span
#         position = torch.arange(seq_length, dtype=torch.long, device=sequence_output.device).unsqueeze(0)
#         relative_pos = position - aspect_pos
#         relative_pos = torch.clamp(relative_pos + self.max_length, 0, 2 * self.max_length)
        
#         rel_pos_embeddings = self.embedding(relative_pos)
#         enhanced_embeddings = sequence_output + rel_pos_embeddings
#         print(enhanced_embeddings)
        
#         adjusted_pos_mask = torch.zeros(enhanced_embeddings.size()[:2], device=enhanced_embeddings.device)
#         adjusted_pos_mask[:, :pos_mask.size(1)] = pos_mask
        
#         aspect_mask = adjusted_pos_mask.unsqueeze(2).float()
#         aspect_embeddings = torch.sum(enhanced_embeddings * aspect_mask, dim=1) / (aspect_mask.sum(dim=1) + 1e-9)
#         aspect_aware_output, _ = self.aspect_attention(enhanced_embeddings.transpose(0, 1),
#                                                        aspect_embeddings.unsqueeze(0),
#                                                        enhanced_embeddings.transpose(0, 1))
#         aspect_aware_output = aspect_aware_output.transpose(0, 1)
        
#         return aspect_aware_output

class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, args, config, vocab_help):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)

        self.pad_id = args.pad_id
        self.max_length = args.max_length
        
        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab, head_vocab = vocab_help
        data = list()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        model_name = 'bert-base-uncased'        
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        
        # Freeze BERT model weights
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text_list = obj['text_list']
            plain_text = obj['text']
    
            tokenized_text = tokenizer(plain_text, padding='max_length', truncation=True, max_length=self.max_length)['input_ids']
            
            aspect = tokenizer(obj['aspect'], padding='max_length', truncation=True, max_length=self.max_length)['input_ids']

            polarity = polarity_dict[obj['label']]
    
            position_mask = pad_sequence(obj['mask'], pad_id=self.pad_id, maxlen=self.max_length, dtype=torch.int64, padding='post', truncating='post')
    
            word_mask = [1 if i < len(obj['mask']) else 0 for i in range(self.max_length)]
    
            aspect_post_start = obj['aspect_post'][0]
            aspect_post_end = obj['aspect_post'][1]
    
            tokens_with_aspect = tokenizer(obj['text']+obj['aspect'], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
            x = self.bert_model(**tokens_with_aspect).last_hidden_state
            if x.shape[1] < args.max_length:
                padding_length = args.max_length - x.shape[1]
                x = torch.nn.functional.pad(x, (0, 0, 0, padding_length))
            x = x.squeeze(0)
            
            data.append({
                'x':x,
                'text': tokenized_text,
                'aspect': aspect,
                'polarity': polarity,
                'pos_mask': position_mask,
                'word_mask': word_mask,
                'aspect_post_start': aspect_post_start,
                'aspect_post_end': aspect_post_end,
                'plain_text': plain_text,
                'text_list': text_list,
            })

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
    
    
    
    
    

# class SentenceDataset(Dataset):
#     ''' PyTorch standard dataset class '''
#     def __init__(self, fname, tokenizer, args, config, vocab_help):
#         self.stop_words = set(stopwords.words('english'))
#         self.punctuation = set(string.punctuation)
        
#         parse = ParseData
#         post_vocab, pos_vocab, dep_vocab, pol_vocab, head_vocab = vocab_help
#         data = list()
#         polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}

#         for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
#             text_list = obj['text_list']
#             plain_text = obj['text']
            
#             text = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['text']]
#             text = pad_sequence(text, pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             aspect = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['aspect']]
#             aspect = pad_sequence(aspect, pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
#             pos = pad_sequence(pos, pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
#             post = pad_sequence(post, pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             head = pad_sequence(obj['head'], pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
#             deprel = pad_sequence(deprel, pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             sentence_length = obj['length']
            
#             polarity = polarity_dict[obj['label']]
            
#             position_mask = pad_sequence(obj['mask'], pad_id=args.pad_id, maxlen=args.max_length, dtype=torch.int64, padding='post', truncating='post')
            
#             word_mask = [1 if i < len(obj['mask']) else 0 for i in range(args.max_length)]
            
#             aspect_post_start = obj['aspect_post'][0]
#             aspect_post_end = obj['aspect_post'][1]
            
#             adj = np.ones(args.max_length) * args.pad_id
            
#             data.append({
#                 'text': text,
#                 'aspect': aspect,
#                 'pos': pos,
#                 'post': post,
#                 'head': head,
#                 'deprel': deprel,
#                 'sentence_length': sentence_length,
#                 'polarity': polarity,
#                 'adj': adj,
#                 'pos_mask': position_mask,
#                 'word_mask': word_mask,
#                 'aspect_post_start': aspect_post_start,
#                 'aspect_post_end': aspect_post_end,
#                 'plain_text': plain_text,
#                 'text_list': text_list,
#             })

#         self._data = data

#     def __getitem__(self, index):
#         return self._data[index]
    
#     def __len__(self):
#         return len(self._data)