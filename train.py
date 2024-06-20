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

import utils.optims as Optim 
import utils.criterion as Criterion 
import utils.lr_scheduler as L
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils.prepare_vocab import VocabHelp
from data_utils.data_utils import SentenceDataset, build_embedding_matrix, build_tokenizer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def save_model(model, path, optimizer, gpus, args,updates=None):
        model_state_dict = model.module.state_dict() if len(gpus) > 1 else model.state_dict()
        checkpoints = {
            'model': model_state_dict,
            'config': args,
            'optim': optimizer,
            'updates': updates}
        torch.save(checkpoints, path)



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
    
    parser.add_argument('--criterion', type=str, default='BCELoss',)
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
    print(tokenizer)
    #embedding matrix
    embedding_matrix = build_embedding_matrix(
            vocab=tokenizer.vocab, 
            embed_dim=args.embed_dim, 
            data_file='{}/{}d_{}_embedding_matrix.dat'.format(args.vocab_dir, str(args.embed_dim), args.dataset))

    print(embedding_matrix)
    logger.info("Loading vocab...")
    
    token_vocab = VocabHelp.load_vocab(args.vocab_dir + '/vocab_tok.vocab')    # token
    
    #train set and test set
    trainset = SentenceDataset(args.dataset_file['train'], tokenizer, opt=args, vocab_help=None)
    testset = SentenceDataset(args.dataset_file['test'], tokenizer, opt=args, vocab_help=None)
                
    # dataloader
    train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size)
    
    # build_model using args and embedding
    
    
    
    
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
    optimizer = Optim.optimizers[args.optimizer](model.parameters(), lr=0.01)
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
    
    # train model

        # eval on test
        
        #f1 score
        
        #save best model
        
        # save_model(model, model_path, optimizer, gpus, args)
        
        # set up logging
        
    
    # draw curves
    
    
    
    
    


if __name__== "__main__":
    main()
         