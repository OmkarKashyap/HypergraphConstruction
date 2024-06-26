import os
import sys
import argparse
import json
import random
import numpy
import logging 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils.optims as Optim 
import utils.criterion as Criterion 
import utils.lr_scheduler as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics

from data_utils.prepare_vocab import VocabHelp
from data_utils.data_utils import SentenceDataset, build_embedding_matrix, build_tokenizer, Seq2Feats

from models.model import HGSCAN

from models.lda_hypergraph import  SemanticHypergraphModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def custom_collate(batch):
    text_batch = [torch.as_tensor(item['text'], dtype=torch.int64) for item in batch]
    # aspect_batch = [torch.as_tensor(item['aspect'], dtype=torch.int64) for item in batch]
    # adj_batch = [torch.as_tensor(item['adj'], dtype=torch.int64) for item in batch]
    # mask_batch = [torch.as_tensor(item['mask'], dtype=torch.int64) for item in batch]
    polarity_batch = [torch.as_tensor(item['polarity'], dtype=torch.int64) for item in batch]
    plain_text = [item['plain_text'] for item in batch]
    
    return {
        'text': text_batch,
        # 'aspect':aspect_batch,
        # 'adj': adj_batch,
        # 'mask': mask_batch,
        'polarity': polarity_batch, 
        'plain_text':plain_text
    }
    
def save_model(model, path, optimizer, gpus, args,updates=None):
        model_state_dict = model.module.state_dict() if len(gpus) > 1 else model.state_dict()
        checkpoints = {
            'model': model_state_dict,
            'config': args,
            'optim': optimizer,
            'updates': updates}
        torch.save(checkpoints, path)

# train model
def train(model, train_dataloader, criterion, optimizer, args, test_dataloader,  embedding_matrix, max_test_acc_overall=0):
    
    max_f1 = 0
    max_test_acc = 0
    step_counter=0
    
    for epoch in range(args.num_epoch):
        logger.info('epoch: {}'.format(epoch))
        
        n_correct, n_total = 0, 0
        for i_batch, batch in enumerate(train_dataloader):
            step_counter+=1
                        
            # inputs = [batch[col] for col in 'text']
            seq2feats = Seq2Feats(embedding_matrix,args)
            x = seq2feats.forward(batch)
            
            targets = batch['polarity']
            targets = torch.tensor([t.item() for t in targets])
            
            # inputs, targets = inputs.to(args.device), targets.to(args.device)
            model.train()
            optimizer.zero_grad()
            
            outputs = model(x)
                       
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            if step_counter % args.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = evaluate(model, test_dataloader, args, embedding_matrix)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            model_path = 'state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format('atae_lstm', 'laptop', test_acc, f1)
                            
                            logger.info('>> saved: {}'.format(model))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
    return max_test_acc, max_f1, model_path
    
def evaluate(model, test_dataloader, args, embedding_matrix, show_results=False):
    model.eval()
    
    n_test_correct, n_test_total = 0,0
    targets_all, outputs_all = None, None
    with torch.no_grad():
        for i_batch, batch in enumerate(test_dataloader):
            # inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            seq2feats = Seq2Feats(embedding_matrix,args)
            x = seq2feats.forward(batch)
            
            targets = batch['polarity']
            targets = torch.tensor([t.item() for t in targets])
            
            outputs = model(x)
            
            n_test_correct += torch.sum(torch.argmax(outputs, -1) == targets).item()
            n_test_total += len(outputs)
            
            targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
            outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
                
    test_acc = n_test_correct / n_test_total
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
    
    labels = targets_all.data.cpu()
    predic = torch.argmax(outputs_all, -1).cpu()
    if show_results:
        report = metrics.classification_report(labels, predic, digits=4)
        confusion = metrics.confusion_matrix(labels, predic)
        return report, confusion, test_acc, f1
    return test_acc, f1

def test(model, test_dataloader, args, embedding_matrix):
        
        model.eval()
        test_report, test_confusion, acc, f1 = evaluate(model, test_dataloader, args, embedding_matrix, show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

def main():
    
    dataset_files = {
        'restaurant': {
            'train': './HypergraphConstruction/dataset/Restaurants_corenlp/train.json',
            'test': './HypergraphConstruction/dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': '../HypergraphConstruction/dataset/Laptops_corenlp/train.json',
            'test': '../HypergraphConstruction/dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './HypergraphConstruction/dataset/Tweets_corenlp/train.json',
            'test': './HypergraphConstruction/dataset/Tweets_corenlp/test.json',
        }
    }
    
    # all variables
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', 
                        help ='Choose cuda is GPU present else cpu')
    
    parser.add_argument('--optimizer', type=str, default='adam',)
                        # help = 'Choose the optimizer: ' + ' | '.join(optim.Optimizer.keys()))
    
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss',)
                        # help = 'Choose the optimizer: ' + ' | '.join(optim.Criterion.keys()))
    
    parser.add_argument('--seed', type=int, default='1234', 
                        help='Set the Random Seed')
    
    parser.add_argument('--gpus', default='', type=str, 
                        help="Use CUDA on the listed devices.")
    
    parser.add_argument('--parallel', action='store_true', 
                        help = 'Use to train on multiple GPUs simultaneously')
    
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--vocab_dir', type=str, default='dataset/Laptops_corenlp')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--log_step', default=5, type=int, help='Logs state after set number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    
    
    parser.add_argument('--n_layers', default=3)
    parser.add_argument('--dropout_rate', default=0.5)
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
    
    
    
    
    args = parser.parse_args()
    
    args.dataset_file = dataset_files[args.dataset]
    
    # device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # set random seed
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    #tokenizer
    tokenizer = build_tokenizer(
            fnames=[args.dataset_file['train'], args.dataset_file['test']], 
            max_length=args.max_length, 
            data_file='{}/{}_tokenizer.dat'.format(args.vocab_dir, args.dataset))
    
    #embedding matrix
    embedding_matrix = build_embedding_matrix(
            vocab=tokenizer.vocab, 
            embed_dim=args.embed_dim, 
            data_file='{}/{}d_{}_embedding_matrix.dat'.format(args.vocab_dir, str(args.embed_dim), args.dataset))

    embedding_matrix = torch.tensor(embedding_matrix)
    
    logger.info("Loading vocab...")
    
    token_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_tok.vocab')   
    post_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
    pos_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
    dep_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
    pol_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polaritytoken_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')    # token
    post_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
    pos_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
    dep_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
    pol_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polarity
    post_size = len(post_vocab)
    pos_size = len(pos_vocab)
    
    vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
    #train set and test set
    trainset = SentenceDataset(args.dataset_file['train'], tokenizer, opt=args, vocab_help=vocab_help)
    testset = SentenceDataset(args.dataset_file['test'], tokenizer, opt=args, vocab_help=vocab_help) 
     
    # dataloader
    train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size, collate_fn=custom_collate)
    
    # build_model using args and embedding
    model = HGSCAN(args)
    # model = SemanticHypergraphModel(args)
    # model = model.to(device)   
    
    
    # number of gpus
    if args.gpus:
        # Split the provided GPUs string into a list
        gpu_ids = args.gpus.split(',')

        # Check availability of each specified GPU
        available_gpus = []
        for gpu_id in gpu_ids:
            if torch.cuda.is_available():
                available_gpus.append(int(gpu_id))
            else:
                print(f"GPU {gpu_id} is not available. Using CPU instead.")

        gpus = available_gpus if available_gpus else []
    else:
        # If no GPUs are specified, check availability of any GPU
        gpus = [torch.device('cuda', i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

    print(f"Selected GPUs: {gpus}")
    
    # criterion/loss function
    criterion = Criterion.criterion[args.criterion]
    
    # optimizer
    optimizer = Optim.optimizers[args.optimizer](model.parameters(),args.learning_rate)
    optimizer.zero_grad()
    # lr scheduler
    scheduler = L.CosineAnnealingLR(optimizer, T_max=50)
    
    # parallel data
    if args.parallel:
        
        gpus = [torch.device('cuda', i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

        if gpus:
            print(f"Using {torch.cuda.device_count()} GPUs: {gpus}")
        else:
            print("No GPUs available. Using CPU.")
    else:
        gpus = []
        print("Not using parallel training.")
        
    if args.parallel and gpus:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model, device_ids=[gpu.index for gpu in gpus])
    
    
    max_test_acc_overall=0
    max_f1_overall=0
    
    max_test_acc, max_f1, model_path = train( model, train_dataloader, criterion, optimizer, args, test_dataloader, embedding_matrix)
    
    logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
    
    max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
    max_f1_overall = max(max_f1, max_f1_overall)
    torch.save(model.state_dict(), model_path)
    
    logger.info('best model saved: {}'.format(model_path))
    logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
    logger.info('max_f1_overall:{}'.format(max_f1_overall))
    test(model, test_dataloader, args, embedding_matrix)


if __name__== "__main__":
    main()
         