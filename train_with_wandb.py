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

import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from ptflops import get_model_complexity_info
from torchviz import make_dot
import networkx as nx
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from proof import calculate_information_content, generate_random_hypergraph

wandb.require("core")
wandb.login()
import warnings
warnings.filterwarnings('ignore')
# wandb.init(project='HCNSCAN-trial', name='training-example')  # Initialize wandb

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = float('-inf')
        self.delta = delta

    def __call__(self, val_f1):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_loss_function(loss_type='cross_entropy', alpha=1, gamma=2):
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        return nn.CrossEntropyLoss()   
    
def quantize_model(model):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized
    


def calculate_flops(model, args,config):
    """
    Calculate FLOPs for a PyTorch model using ptflops library.
    
    Args:
    model (torch.nn.Module): The PyTorch model to analyze.
    input_size (tuple): The size of the input tensor, excluding batch size.
                        For example, for an image input it might be (3, 224, 224).
    
    Returns:
    int: The number of FLOPs. If FLOP calculation fails, returns the number of parameters.
    """
    input_size = (config.batch_size, args.max_length, 768)
    assert isinstance(model, torch.nn.Module), "Model must be a PyTorch nn.Module"
    assert isinstance(input_size, tuple), "Input size must be a tuple"
    
    model.eval()  # Set the model to evaluation mode
    
    try:
        flops, _ = get_model_complexity_info(
            model, input_size, 
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        return int(flops)
        
    except Exception as e:
        print(f"Error in FLOP calculation: {str(e)}")
        print("Falling back to parameter count.")
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def custom_collate(batch):
    device = 'cpu' if torch.cuda.is_available() else 'cpu'

    x_batch = [torch.as_tensor(item['x'], dtype=torch.float32).to(device) for item in batch]
    x_batch = torch.stack(x_batch)

    text_batch = [torch.as_tensor(item['text'], dtype=torch.int64).to(device) for item in batch]
    aspect_batch = [torch.as_tensor(item['aspect'], dtype=torch.int64).to(device) for item in batch]
    polarity_batch = [torch.as_tensor(item['polarity'], dtype=torch.int64).to(device) for item in batch]
    
    position_mask_batch = [torch.as_tensor(item['pos_mask'], dtype=torch.int64).to(device) for item in batch]
    word_mask_batch = [torch.as_tensor(item['word_mask'], dtype=torch.int64).to(device) for item in batch]
    aspect_post_start_batch = [torch.as_tensor(item['aspect_post_start'], dtype=torch.int64).to(device) for item in batch]
    aspect_post_end_batch = [torch.as_tensor(item['aspect_post_end'], dtype=torch.int64).to(device) for item in batch]
    plain_text_batch = [item['plain_text'] for item in batch]
    text_list_batch = [item['text_list'] for item in batch]

    return {
        'x': x_batch,
        'text': text_batch,
        'aspect': aspect_batch,
        'polarity': polarity_batch,
        'pos_mask': position_mask_batch,
        'word_mask': word_mask_batch,
        'aspect_post_start': aspect_post_start_batch,
        'aspect_post_end': aspect_post_end_batch,
        'plain_text': plain_text_batch,
        'text_list': text_list_batch,
    }

def save_model(model, path, optimizer, gpus, args, updates=None):
    model_state_dict = model.module.state_dict() if len(gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': args,
        'optim': optimizer,
        'updates': updates}
    torch.save(checkpoints, path)
    

def train(model, train_dataloader, criterion, optimizer, scheduler, args, config, test_dataloader, epoch, max_test_acc_overall=0, max_f1=0, max_test_acc=0, step_counter=0):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.train()
    n_correct, n_total = 0, 0
    epoch_train_loss = 0
    
    # For gradient variance tracking
    layer_gradients = {name: [] for name, _ in model.named_parameters()}
    
    # For activation statistics
    activation_stats = {}
    
    # For gradient variance
    all_gradients = []
    # For mutual information
    all_activations = []

    for i_batch, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
        step_counter += 1
        optimizer.zero_grad()

        x = prepare_input_data(batch, args, config)
        
        outputs = model(x)
        outputs = outputs.squeeze(0)
        
        targets = batch['polarity'].to(device)
        
        main_loss = criterion(outputs, targets)  
              
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)    
                
        loss = main_loss + config.l2_lambda * l2_reg

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        
         # Track gradient variance and norms
        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        all_gradients.append(grad.detach().cpu().numpy())
        
        # Track gradient variance and norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                layer_gradients[name].append(grad_norm)
                wandb.log({f'gradient_norm/{name}': grad_norm, 'step': step_counter})
        
        optimizer.step()
        scheduler.step()

        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        n_total += len(outputs)
        epoch_train_loss += loss.item()
        
        # Collect activations for mutual information
        all_activations.append(outputs.detach().cpu().numpy())

        # Log activation statistics (for the first batch only to save computation)
        if i_batch == 0:
            for name, module in model.named_modules():
                if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                    activation_stats[name] = outputs.detach().cpu().numpy()
                    
    # Compute gradient variance
    all_gradients = np.stack(all_gradients)
    grad_var = np.trace(np.cov(all_gradients.T))
    wandb.log({'gradient_variance': grad_var, 'epoch': epoch})

    # Compute mutual information
    all_activations = np.concatenate(all_activations)
    entropy = -np.sum(all_activations * np.log(all_activations + 1e-10), axis=1).mean()
    mutual_info = -entropy
    wandb.log({'mutual_information': mutual_info, 'epoch': epoch})

    # Log gradient variance
    for name, grad_list in layer_gradients.items():
        if grad_list:
            wandb.log({f'gradient_variance/{name}': np.var(grad_list), 'epoch': epoch})
    
    # Log activation statistics
    for name, activations in activation_stats.items():
        wandb.log({
            f'activation_mean/{name}': np.mean(activations),
            f'activation_std/{name}': np.std(activations),
            'epoch': epoch
        })
    
    # Evaluate
    test_acc, test_loss, f1 = evaluate(model, test_dataloader, criterion, args, config)
    
    wandb.log({
        'train_loss': epoch_train_loss / len(train_dataloader),
        'train_accuracy': n_correct / n_total,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'epoch': epoch
    })

    # Log weight distribution
    for name, param in model.named_parameters():
        if 'weight' in name:
            wandb.log({f'weight_dist/{name}': wandb.Histogram(param.detach().cpu().numpy()), 'epoch': epoch})

    return test_acc, f1

def evaluate(model, test_dataloader, criterion, args, config, show_results=False):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.eval()

    n_correct, n_total = 0, 0
    total_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            x = prepare_input_data(batch, args, config)
            
            outputs = model(x)
            outputs = outputs.squeeze(0)
            targets = batch['polarity'].to(device)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, -1)
            n_correct += (predictions == targets).sum().item()
            n_total += len(targets)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    test_acc = n_correct / n_total
    test_loss = total_loss / len(test_dataloader)
    f1 = metrics.f1_score(all_targets, all_predictions, average='macro')
    
    if show_results:
        report = metrics.classification_report(all_targets, all_predictions, digits=4)
        confusion = metrics.confusion_matrix(all_targets, all_predictions)
        return report, confusion, test_acc, f1
    
    return test_acc, test_loss, f1

def train_and_evaluate(config=None, ):
    
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        trainset = SentenceDataset(args.dataset_file['train'], args, config)
        testset = SentenceDataset(args.dataset_file['test'], args, config)
        train_dataloader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate)
        test_dataloader = DataLoader(dataset=testset, batch_size=config.batch_size, collate_fn=custom_collate)
        
        model = HGSCAN(args, config)
        model.to(device)

        optimizer = Optim.optimizers[config.optim](model.parameters(), config.learning_rate)
        
         # Initialize the scheduler
        num_training_steps = config.epochs * len(train_dataloader)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warm-up
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        # Toggle between Focal Loss and Cross Entropy
        criterion = get_loss_function(config.loss_type, alpha=config.focal_alpha, gamma=config.focal_gamma)
        criterion.to(device)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
        
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)      
        
        # For stability analysis
        all_final_accuracies = []
        all_final_f1_scores = []
        
        # For convergence speed comparison
        performance_threshold = config.performance_threshold 
        epochs_to_threshold = []

        # For learning rate vs. performance
        lr_performance_data = []

        # For training time per epoch
        epoch_times = []

        start_time = time.time()

        for epoch in range(config.epochs):
            epoch_start_time = time.time()
            max_test_acc, max_f1 = train(model, train_dataloader, criterion, optimizer, scheduler,  args, config, test_dataloader, epoch)
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Early stopping check
            if early_stopping(max_f1):
                print("Early stopping")
                break
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': current_lr, 'epoch': epoch})
            
            # Performance vs. Model Size
            model_size = sum(p.numel() for p in model.parameters())
            wandb.log({
                'model_size_vs_performance': wandb.plot.scatter(
                    "Model Size vs. Performance",
                    {"model_size": model_size, "f1_score": max_f1}
                ),
                'epoch': epoch
            })

            # Convergence Speed
            if max_test_acc >= performance_threshold and not epochs_to_threshold:
                epochs_to_threshold.append(epoch + 1)

            # Learning Rate vs. Performance
            lr_performance_data.append((current_lr, epoch, max_test_acc))

            # Log epoch time
            wandb.log({'epoch_time': epoch_time, 'epoch': epoch})
        
            # Log eigenvalue spectrum
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() == 2:
                    _, s, _ = torch.svd(param)
                    wandb.log({f'eigenvalue_spectrum/{name}': wandb.Histogram(s.cpu().numpy()), 'epoch': epoch})

            # Log hypergraph structure metrics
            hypergraph_metrics = calculate_hypergraph_metrics(model)  # You need to implement this function
            wandb.log(hypergraph_metrics)

            # Log attention weights (if applicable)
            if hasattr(model, 'attention_weights'):
                attention_weights = model.attention_weights.detach().cpu().numpy()
                wandb.log({'attention_weights': wandb.Image(plt.imshow(attention_weights, cmap='viridis'))})
        
        # Computational Efficiency
        end_time = time.time()
        total_time = end_time - start_time
        
        flops = calculate_flops(model, args, config)  
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            'computational_efficiency': wandb.plot.scatter(
                "Performance vs. Computational Cost",
                {"flops": flops, "time": total_time, "f1_score": max_f1, 'params': params}
            )
        })

        # Training Stability
        all_final_accuracies.append(max_test_acc)
        all_final_f1_scores.append(max_f1)

        # Final evaluation
        final_report, final_confusion, final_acc, final_f1 = evaluate(model, test_dataloader, criterion, args, config, show_results=True)
        wandb.log({
            'final_test_accuracy': final_acc,
            'final_f1_score': final_f1,
            'classification_report': wandb.Table(data=[final_report], columns=["Metric"])
        })
        

        # Log additional plots
        log_stability_plot(all_final_accuracies, all_final_f1_scores)
        log_convergence_speed_plot(epochs_to_threshold)
        log_lr_performance_heatmap(lr_performance_data)
        log_training_time_plot(epoch_times)
        
        # Log loss landscape
        loss_landscape = calculate_loss_landscape(model, train_dataloader, criterion) 
        wandb.log({'loss_landscape': wandb.Image(loss_landscape)})

        # Log hyperedge analysis
        hyperedge_sizes = analyze_hyperedges(model) 
        wandb.log({'hyperedge_size_distribution': wandb.Histogram(hyperedge_sizes)})

        # Log performance on different data subsets
        subset_performance = evaluate_on_subsets(model, test_dataloader, criterion)
        for subset, perf in subset_performance.items():
            wandb.log({f'subset_performance/{subset}': perf})

        # Log hypergraph sparsity
        sparsity = calculate_hypergraph_sparsity(model) 
        wandb.log({'hypergraph_sparsity': sparsity})
        
        return max_test_acc, max_f1

def prepare_input_data(batch, args, config):

    x_complete = [
        batch['x'],
        batch['text'],
        batch['aspect'],
        batch['pos_mask'],
        batch['word_mask'],
        batch['aspect_post_start'],
        batch['aspect_post_end'],
        batch['plain_text'],
        batch['text_list'],
    ]

    return x_complete

def log_stability_plot(accuracies, f1_scores):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[accuracies, f1_scores], inner="box")
    plt.xticks([0, 1], ['Accuracy', 'F1 Score'])
    plt.title('Distribution of Final Performance Across Runs')
    wandb.log({"stability_plot": wandb.Image(plt)})
    plt.close()

def log_convergence_speed_plot(epochs_to_threshold):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=range(len(epochs_to_threshold)), y=epochs_to_threshold)
    plt.xlabel('Run')
    plt.ylabel('Epochs to Reach Threshold')
    plt.title('Convergence Speed Comparison')
    wandb.log({"convergence_speed_plot": wandb.Image(plt)})
    plt.close()

def log_lr_performance_heatmap(lr_performance_data):
    lr_data = np.array(lr_performance_data)
    lr_values = np.unique(lr_data[:, 0])
    epoch_values = np.unique(lr_data[:, 1])
    performance_matrix = np.zeros((len(lr_values), len(epoch_values)))

    for lr, epoch, performance in lr_data:
        i = np.where(lr_values == lr)[0][0]
        j = np.where(epoch_values == epoch)[0][0]
        performance_matrix[i, j] = performance

    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_matrix, xticklabels=epoch_values, yticklabels=lr_values, cmap='viridis')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Performance Heatmap')
    wandb.log({"lr_performance_heatmap": wandb.Image(plt)})
    plt.close()

def log_training_time_plot(epoch_times):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_times) + 1), epoch_times)
    plt.xlabel('Epoch')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time per Epoch')
    wandb.log({"training_time_plot": wandb.Image(plt)})
    plt.close()

def calculate_hypergraph_metrics(model):
    # Assuming model.hypergraph is a tensor representing the hypergraph
    H = model.hypergraph.detach().cpu().numpy()
    
    # Average node degree
    node_degrees = np.sum(H, axis=1)
    avg_node_degree = np.mean(node_degrees)
    
    # Clustering coefficient (approximation for hypergraphs)
    G = nx.from_numpy_array(np.dot(H, H.T))
    clustering_coeff = nx.average_clustering(G)
    
    # Hypergraph density
    num_nodes, num_edges = H.shape
    max_possible_edges = 2**num_nodes - 1
    density = num_edges / max_possible_edges
    
    return {
        'avg_node_degree': avg_node_degree,
        'clustering_coefficient': clustering_coeff,
        'hypergraph_density': density
    }

def calculate_feature_importance(model):
    # Assuming model.feature_weights is a tensor of feature weights
    feature_weights = model.feature_weights.detach().cpu().numpy()
    
    # Normalize weights
    feature_importance = np.abs(feature_weights) / np.sum(np.abs(feature_weights))
    
    return dict(enumerate(feature_importance))

def calculate_loss_landscape(model, dataloader, criterion):
    # Choose two random directions in parameter space
    params = [p for p in model.parameters() if p.requires_grad]
    direction1 = [torch.randn_like(p) for p in params]
    direction2 = [torch.randn_like(p) for p in params]
    
    # Normalize directions
    d1_norm = torch.sqrt(sum(torch.sum(d**2) for d in direction1))
    d2_norm = torch.sqrt(sum(torch.sum(d**2) for d in direction2))
    direction1 = [d/d1_norm for d in direction1]
    direction2 = [d/d2_norm for d in direction2]
    
    # Calculate loss landscape
    alpha_range = np.linspace(-1, 1, 20)
    beta_range = np.linspace(-1, 1, 20)
    loss_landscape = np.zeros((len(alpha_range), len(beta_range)))
    
    original_params = [p.clone() for p in params]
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Update model parameters
            for p, d1, d2, orig in zip(params, direction1, direction2, original_params):
                p.data = orig + alpha*d1 + beta*d2
            
            # Compute loss
            total_loss = 0
            for batch in dataloader:
                outputs = model(batch)
                loss = criterion(outputs, batch['labels'])
                total_loss += loss.item()
            loss_landscape[i, j] = total_loss / len(dataloader)
    
    # Reset model parameters
    for p, orig in zip(params, original_params):
        p.data = orig
    
    # Plot loss landscape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha_range, beta_range)
    surf = ax.plot_surface(X, Y, loss_landscape, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    plt.colorbar(surf)
    
    return fig

def analyze_hyperedges(model):
    # Assuming model.hypergraph is a tensor representing the hypergraph
    H = model.hypergraph.detach().cpu().numpy()
    
    hyperedge_sizes = np.sum(H, axis=0)
    
    return hyperedge_sizes

def evaluate_on_subsets(model, dataloader, criterion):
    # Define subsets (example: by class)
    subsets = {}
    for batch in dataloader:
        for input, label in zip(batch['inputs'], batch['labels']):
            if label.item() not in subsets:
                subsets[label.item()] = []
            subsets[label.item()].append(input)
    
    results = {}
    for subset_name, subset_data in subsets.items():
        subset_loader = torch.utils.data.DataLoader(subset_data, batch_size=32)
        subset_acc, subset_loss, subset_f1 = evaluate(model, subset_loader, criterion)
        results[f'Subset_{subset_name}'] = {
            'accuracy': subset_acc,
            'loss': subset_loss,
            'f1_score': subset_f1
        }
    
    return results

def calculate_hypergraph_sparsity(model):
    # Assuming model.hypergraph is a tensor representing the hypergraph
    H = model.hypergraph.detach().cpu().numpy()
    
    total_elements = H.size
    non_zero_elements = np.count_nonzero(H)
    
    sparsity = 1 - (non_zero_elements / total_elements)
    
    return sparsity

def main():
    # wandb.init(project='HCNSCAN-trial', name='training-example')  # Initialize wandb
    sweep_config = {
        'method': 'bayes',
        'parameters': {
            'optim' : {
                'values' : ['rmsprop']
            },
            'epochs' : {
                'values' : [30]
            },
            'learning_rate': {'min': 1e-5, 'max': 1e-2},
            'batch_size': {
                'values': [64]
            },
            'dropout_rate': {
                'values': [0.1, 0.2, 0.3]
            },
            'top_k_knn':{
                'values' : [2,3,4]
            },
            'top_k_lda':{
                'values' : [2,3,4]
            },
            'performance_threshold':{
                'values' : [0.2, 0.5, 0.8]
            }, 
            'loss_type': {
                'values': ['cross_entropy', 'focal']
            },
            'focal_alpha':{
                'values' : [0.25, 0.5]
            }, 
            'focal_gamma':{
                'values' : [1,2,3,4]
            }, 
            'early_stopping_patience': {
                'values' : [5,10]
            },
            'l2_lambda': {
                'min': 1e-5,
                'max': 1e-3,
                'distribution': 'log_uniform'
            },
            'max_grad_norm': {
                    'min': 1.0,
                    'max': 10.0
            },
            
            
            'min_samples': {
                'values' : [3, 4, 5]
            },
            'eps' : {
                'values' : [0.001, 0.01 , 0.05, 0.1]
            },
            
            'num_topics' : {
                'values' : [30, 40, 50]
            },
            'n_layers' : {
                'values' : [2, 3]
            },
            
            
        },
        'metric': {
            'name': 'f1_score',
            'goal': 'maximize'
        }
    }

    
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
    parser.add_argument('--device', type=str, default='cpu', help='Choose cuda if GPU present else cpu')
    parser.add_argument('--optimizer', type=str, default='adam', help='Choose the optimizer')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='Choose the criterion')
    parser.add_argument('--embedding_name', type=str, default='glove')
    parser.add_argument('--seed', type=int, default=1234, help='Set the Random Seed')
    parser.add_argument('--gpus', default='', type=str, help='Use CUDA on the listed devices.')
    parser.add_argument('--parallel', action='store_true', help='Use to train on multiple GPUs simultaneously')
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--extra_padding', default=0, type=int)
    parser.add_argument('--vocab_dir', type=str, default='dataset/Laptops_corenlp')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--pad_id', default=-1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--log_step', default=5, type=int, help='Logs state after set number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--n_layers', default=2)
    parser.add_argument('--dropout_rate', default=0.3)
    parser.add_argument('--eps', default=0.01)
    parser.add_argument('--min_samples', default=3)
    parser.add_argument('--output_size', default=10)
    parser.add_argument('--dim_in', default=768)
    parser.add_argument('--hidden_num', default=5)
    parser.add_argument('--ft_dim', default=300)
    parser.add_argument('--n_categories', default=3)
    parser.add_argument('--has_bias', type=str, default=True)
    parser.add_argument('--num_topics', default=10)
    parser.add_argument('--top_k', default=5)

    global args
    args = parser.parse_args()
    args.dataset_file = dataset_files[args.dataset]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # wandb.agent(sweep_id, train_and_evaluate)
    sweep_id = wandb.sweep(sweep_config, project="HCNSCAN-KNN-trial")
    wandb.agent(sweep_id, train_and_evaluate, count=600)

    # hyperparameter_tuning(args)

if __name__ == "__main__":
    main()