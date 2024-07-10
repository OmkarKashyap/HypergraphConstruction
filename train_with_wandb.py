import os
import sys
import argparse
import json
import random
import numpy
import logging
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from tqdm import tqdm

from transformers import BertTokenizer, BertModel

import utils.optims as Optim
import utils.criterion as Criterion
import utils.lr_scheduler as L
from sklearn import metrics

from data_utils.prepare_vocab import VocabHelp
from data_utils.data_utils import SentenceDataset, build_embedding_matrix, build_tokenizer, Seq2Feats

from models.model import HGSCAN
from models.lda_hypergraph import SemanticHypergraphModel
from models.omk_dep_hg.dep_hg import DependencyHG


wandb.require("core")
wandb.login()
import warnings
warnings.filterwarnings('ignore')
# wandb.init(project='HCNSCAN-trial', name='training-example')  # Initialize wandb

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def pad_incidence_matrix(inc_mat, max_edges):
    current_edges = inc_mat.size(1)
    if current_edges < max_edges:
        padding_size = max_edges - current_edges
        padding = torch.zeros(inc_mat.size(0), padding_size)
        inc_mat = torch.cat((inc_mat, padding), dim=1)
    return inc_mat

def custom_collate(batch):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    text_batch = [torch.as_tensor(item['text'], dtype=torch.int64).to(device) for item in batch]
    aspect_batch = [torch.as_tensor(item['aspect'], dtype=torch.int64).to(device) for item in batch]
    pos_batch = [torch.as_tensor(item['pos'], dtype=torch.int64).to(device) for item in batch]
    post_batch = [torch.as_tensor(item['post'], dtype=torch.int64).to(device) for item in batch]
    head_batch = [torch.as_tensor(item['head'], dtype=torch.int64).to(device) for item in batch]
    deprel_batch = [torch.as_tensor(item['deprel'], dtype=torch.int64).to(device) for item in batch]
    sentence_length_batch = [torch.as_tensor(item['sentence_length'], dtype=torch.int64).to(device) for item in batch]
    polarity_batch = [torch.as_tensor(item['polarity'], dtype=torch.int64).to(device) for item in batch]
    adj_batch = [torch.as_tensor(item['adj'], dtype=torch.int64).to(device) for item in batch]
    position_mask_batch = [torch.as_tensor(item['pos_mask'], dtype=torch.int64).to(device) for item in batch]
    word_mask_batch = [torch.as_tensor(item['word_mask'], dtype=torch.int64).to(device) for item in batch]
    aspect_post_start_batch = [torch.as_tensor(item['aspect_post_start'], dtype=torch.int64).to(device) for item in batch]
    aspect_post_end_batch = [torch.as_tensor(item['aspect_post_end'], dtype=torch.int64).to(device) for item in batch]
    plain_text_batch = [item['plain_text'] for item in batch]
    text_list_batch = [item['text_list'] for item in batch]

    batch.sort(key=lambda x: x['incidence_matrix'].size(1), reverse=True)
    max_edges = batch[0]['incidence_matrix'].size(1)
    padded_tensors = []

    for sample in batch:
        padded_tensors.append(pad_incidence_matrix(sample['incidence_matrix'], max_edges).to(device))

    padded_tensors = torch.stack(padded_tensors)

    return {
        'text': text_batch,
        'aspect': aspect_batch,
        'pos': pos_batch,
        'post': post_batch,
        'head': head_batch,
        'deprel': deprel_batch,
        'sentence_length': sentence_length_batch,
        'polarity': polarity_batch,
        'adj': adj_batch,
        'pos_mask': position_mask_batch,
        'word_mask': word_mask_batch,
        'aspect_post_start': aspect_post_start_batch,
        'aspect_post_end': aspect_post_end_batch,
        'plain_text': plain_text_batch,
        'text_list': text_list_batch,
        'incidence_matrix': padded_tensors
    }

def save_model(model, path, optimizer, gpus, args, updates=None):
    model_state_dict = model.module.state_dict() if len(gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': args,
        'optim': optimizer,
        'updates': updates}
    torch.save(checkpoints, path)

def train(model, train_dataloader, criterion, optimizer, args, test_dataloader, embedding_matrix, epoch, max_test_acc_overall=0, max_f1 = 0, max_test_acc = 0,step_counter = 0 ):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.train()
    n_correct, n_total = 0, 0
    epoch_train_loss = 0


    for i_batch, batch in enumerate(train_dataloader):
        step_counter += 1

        if args.embedding_name == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokens = tokenizer(batch['plain_text'], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            x = bert_model(**tokens).last_hidden_state
            if x.shape[1] < args.max_length:
                padding_length = args.max_length - x.shape[1]
                x = torch.nn.functional.pad(x, (0, 0, 0, padding_length))
        elif args.embedding_name == 'glove':
            seq2feats = Seq2Feats(embedding_matrix, args)
            x = seq2feats.forward(batch)

        x_complete = [
            x,
            batch['text'],
            batch['aspect'],
            batch['pos'],
            batch['post'],
            batch['head'],
            batch['deprel'],
            batch['sentence_length'],
            batch['adj'],
            batch['pos_mask'],
            batch['word_mask'],
            batch['aspect_post_start'],
            batch['aspect_post_end'],
            batch['plain_text'],
            batch['text_list'],
        ]

        targets = batch['polarity']
        targets = torch.tensor([t.item() for t in targets]).to(device)

        optimizer.zero_grad()
        outputs = model(x_complete, batch['incidence_matrix'])
        outputs = outputs.squeeze(0)
        loss = criterion(outputs, targets)

        # Calculate L2 regularization
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)**2
            
        loss += args.lambda_reg * l2_reg

        loss.backward()
        optimizer.step()

        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        n_total += len(outputs)
        epoch_train_loss += loss.item()

        if step_counter % args.log_step == 0:
            test_acc, test_loss, f1 = evaluate(model, test_dataloader, criterion, args, embedding_matrix)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                if test_acc > max_test_acc_overall:
                    if not os.path.exists('state_dict'):
                        os.mkdir('state_dict')
                    model_path = 'state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format('Semantic_HG', 'laptop', test_acc, f1)

            if f1 > max_f1:
                max_f1 = f1

            

            logger.info('loss: {:.4f}, acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(
                loss.item(), n_correct / n_total, test_loss, test_acc, f1))

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    avg_train_acc = n_correct / n_total
    logger.info('Epoch {} average metrics: Train Loss {:.4f}, Train Acc {:.4f}, Test Acc {:.4f}, F1 {:.4f}'.format(
        epoch, avg_train_loss, avg_train_acc, test_acc, f1))
    # Log metrics to wandb  
    # wandb.log({
    #             'train_loss': epoch_train_loss / (i_batch + 1),
    #             'train_accuracy': n_correct / n_total,
    #             'test_loss': test_loss,
    #             'test_accuracy': test_acc,
    #             'f1_score': f1,
    #             'epoch': epoch
    #         })
    

    return max_test_acc, max_f1, model_path


def evaluate(model, test_dataloader, criterion, args, embedding_matrix, show_results=False):
    model.eval()

    n_test_correct, n_test_total = 0, 0
    targets_all, outputs_all = None, None
    total_loss = 0

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i_batch, batch in enumerate(test_dataloader):
            if args.embedding_name == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                tokens = tokenizer(batch['plain_text'], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
                bert_model = BertModel.from_pretrained('bert-base-uncased')
                x = bert_model(**tokens).last_hidden_state
                if x.shape[1] < args.max_length:
                    padding_length = args.max_length - x.shape[1]
                    x = torch.nn.functional.pad(x, (0, 0, 0, padding_length))

            elif args.embedding_name == 'glove':
                seq2feats = Seq2Feats(embedding_matrix, args)
                x = seq2feats.forward(batch)

            x_complete = [
                x,
                batch['text'],
                batch['aspect'],
                batch['pos'],
                batch['post'],
                batch['head'],
                batch['deprel'],
                batch['sentence_length'],
                batch['adj'],
                batch['pos_mask'],
                batch['word_mask'],
                batch['aspect_post_start'],
                batch['aspect_post_end'],
                batch['plain_text'],
                batch['text_list'],
            ]

            targets = batch['polarity']
            targets = torch.tensor([t.item() for t in targets]).to(device)

            outputs = model(x_complete, batch['incidence_matrix'])
            outputs = outputs.squeeze(0)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            n_test_correct += torch.sum(torch.argmax(outputs, -1) == targets).item()
            n_test_total += len(outputs)

            targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
            outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs

    test_acc = n_test_correct / n_test_total
    test_loss = total_loss / len(test_dataloader)
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

    labels = targets_all.data.cpu()
    predic = torch.argmax(outputs_all, -1).cpu()
    if show_results:
        report = metrics.classification_report(labels, predic, digits=4)
        confusion = metrics.confusion_matrix(labels, predic)
        return report, confusion, test_acc, f1
    return test_acc, test_loss, f1

def train_and_evaluate(config=None):
    
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        tokenizer = build_tokenizer(
            fnames=[args.dataset_file['train'], args.dataset_file['test']],
            max_length=args.max_length,
            data_file='{}/{}_tokenizer.dat'.format(args.vocab_dir, args.dataset))

        embedding_matrix = build_embedding_matrix(
            vocab=tokenizer.vocab,
            embed_dim=args.embed_dim,
            data_file='{}/{}d_{}_embedding_matrix.dat'.format(args.vocab_dir, str(args.embed_dim), args.dataset))

        embedding_matrix = torch.tensor(embedding_matrix)
        logger.info("Loading vocab...")
        token_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_tok.vocab')
        post_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_post.vocab')
        pos_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_pos.vocab')
        dep_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_dep.vocab')
        pol_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_pol.vocab')
        head_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_head.vocab')
        logger.info("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))
        args.vocab_size = len(token_vocab) + len(post_vocab) + len(pos_vocab) + len(dep_vocab) + len(pol_vocab) + len(head_vocab)
        vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab, head_vocab)

        trainset = SentenceDataset(args.dataset_file['train'], tokenizer, args, config, vocab_help, embedding_matrix, f'pickled_datasets/train_{config.eps}_{config.min_samples}.pkl')
        testset = SentenceDataset(args.dataset_file['test'], tokenizer, args, config, vocab_help, embedding_matrix, f'pickled_datasets/test_{config.eps}_{config.min_samples}.pkl')
        train_dataloader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate)
        test_dataloader = DataLoader(dataset=testset, batch_size=config.batch_size, collate_fn=custom_collate)

        embedding_matrix_copy = embedding_matrix.clone()
        model = HGSCAN(args, config)

        optimizer = Optim.optimizers[config.optim](model.parameters(), config.learning_rate)
        criterion = Criterion.criterion[args.criterion]

        model.to(device)
        criterion.to(device)


        for epoch in tqdm(range(config.epochs)):
            max_test_acc, max_f1, model_path = train(model, train_dataloader, criterion, optimizer, args, test_dataloader, embedding_matrix, epoch)
            wandb.log({'f1_score': max_f1})

        return max_test_acc, max_f1


def test(model, test_dataloader, args, embedding_matrix):
    model.eval()
    test_report, test_confusion, acc, f1 = evaluate(model, test_dataloader, args, embedding_matrix, show_results=True)
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)

def main():
    # wandb.init(project='HCNSCAN-trial', name='training-example')  # Initialize wandb
    sweep_config = {
        'method': 'bayes',
        'parameters': {
            'optim' : {
                'values' : ['rmsprop']
            },
            'epochs' : {
                'values' : [30, 40]
            },
            'learning_rate': {
                'values': [ 0.0001, 0.00005, 0.00001]
            },
            'batch_size': {
                'values': [64]
            },
            'dropout_rate': {
                'values': [0.3]
            },
            'min_samples': {
                'values' : [4]
            },
            'eps' : {
                'values' : [0.01]
            },
            'top_k':{
                'values' : [4,5]
            },
            'num_topics' : {
                'values' : [50, 60, 70]
            },
            'n_layers' : {
                'values' : [5,4, 3, 2]
            },
            'gnn_aggregation_type' : {
                'values' : ['mean', 'sum', 'max', 'attention']
            },
            # 'attention_heads' : {
            #     'values' : [8,4,2,1]
            # },
            
        },
        'metric': {
            'name': 'f1_score',
            'goal': 'maximize'
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="HCNSCAN-with-norm-and-aspect")
    
    dataset_files = {
        'restaurant': {
            'train': '../HypergraphConstruction/dataset/Restaurants_corenlp/train.json',
            'test': '../HypergraphConstruction/dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': '../HypergraphConstruction/dataset/Laptops_corenlp/train.json',
            'test': '../HypergraphConstruction/dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': '../HypergraphConstruction/dataset/Tweets_corenlp/train.json',
            'test': '../HypergraphConstruction/dataset/Tweets_corenlp/test.json',
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Choose cuda if GPU present else cpu')
    parser.add_argument('--optimizer', type=str, default='adam', help='Choose the optimizer')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='Choose the criterion')
    parser.add_argument('--embedding_name', type=str, default='glove')
    parser.add_argument('--seed', type=int, default=1234, help='Set the Random Seed')
    parser.add_argument('--gpus', default='', type=str, help='Use CUDA on the listed devices.')
    parser.add_argument('--parallel', action='store_true', help='Use to train on multiple GPUs simultaneously')
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--max_length', default=90, type=int)
    parser.add_argument('--extra_padding', default=0, type=int)
    parser.add_argument('--vocab_dir', type=str, default='dataset/Laptops_corenlp')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--pad_id', default=-1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--log_step', default=5, type=int, help='Logs state after set number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--n_layers', default=2)
    parser.add_argument('--dropout_rate', default=0.3)
    parser.add_argument('--eps', default=0.01)
    parser.add_argument('--min_samples', default=3)
    parser.add_argument('--output_size', default=10)
    parser.add_argument('--dim_in', default=300)
    parser.add_argument('--hidden_num', default=5)
    parser.add_argument('--ft_dim', default=300)
    parser.add_argument('--n_categories', default=3)
    parser.add_argument('--has_bias', type=str, default=True)
    parser.add_argument('--num_topics', default=10)
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--tensorboard_log_dir', type=str, default='model_state_params')

    parser.add_argument('--gnn_aggregation_type', choices=['sum', 'mean', 'max', 'attention'], default='attention')
    parser.add_argument('--attention_heads', default=4)
    parser.add_argument('--lambda_reg', default=0.001)

    parser.add_argument('--louvain_threshold', default=0.5)
    parser.add_argument('--louvain_max_communities', default=5)
    parser.add_argument('--girvan_newman_threshold', default=0.5)
    parser.add_argument('--label_propogation_threshold', default=0.5)
    parser.add_argument('--kernighan_lin_threshold', default=0.5)

    parser.add_argument('--pos_dim', type=int, default=0)
    parser.add_argument('--ner_dim', type=int, default=300)
    parser.add_argument('--prune_k', type=int, default=-1)

    parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')
    parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
    parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
    parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--rnn', action='store_true', default='false', help='Flag to use RNN')
    parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
    parser.add_argument('--rnn_mem_dim', default=300, help='Dimension of rnn output. More values means more complex but more computation')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    global args
    args = parser.parse_args()
    args.dataset_file = dataset_files[args.dataset]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # wandb.agent(sweep_id, train_and_evaluate)
    wandb.agent(sweep_id, train_and_evaluate, count=600)

    # hyperparameter_tuning(args)

if __name__ == "__main__":
    main()


