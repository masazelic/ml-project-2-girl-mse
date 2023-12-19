# This code is implemented by Marija Zelic, Sara Zatezalo and Elena Mrdja for the purposes of the ML4Science project

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, SubsetRandomSampler
import pandas as pd
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from types import SimpleNamespace
from utils import data, evaluation, models, helpers
from sklearn.model_selection import KFold
import torch.nn.functional as F 

# Paths for the auxiliary task
TRAIN_PATH = "./data/all_train.csv"
TEST_PATH = "./data/all_test.csv"

# Paths for the classification task (change depending on balancing or consensus sequences)
# all_off_sequences_v2.txt and all_on_sequences_v2.txt for balancing
# off_sequences_v2.txt and off_sequences_v2.txt for consensus
OFF_SEQUENCES_PATH = "./data/all_off_sequences_v2.txt"
ON_SEQUENCES_PATH = "./data/all_on_sequences_v2.txt"

PROT_LENGTH = 153
MAX_PROT_MASK = 50
BATCH_SIZE = 512
BATCH_SIZE_CONSENSUS = 8
BATCH_SIZE_BALANCED = 16

def set_seed(seed=42):
    """
    Set random seed.
    
    Args:
        seed (int, optional): Random seed. Defaults to 42.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_auxiliary_task(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Function that fine-tunes BERT model for MLM problem on protein sequences.
    
    Args:
        train_path (str): Path to the .csv file containing train sequences.
        test_path (str): Path to the .csv file containing test sequences.
    
    Returns:
        model (nn.Module): Trained model.
    """
    # Set device and seed
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define parameters for tokenization 
    kmer = 2 # how many consecutive amino acids is considered as one token
    mask_length = 2 # how many consecutive tokens is masked
    
    # Amino acid vocabulary
    vocab = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "S", "P", "T", "W", "Y", "V", "-"]
    VOCAB_2MER = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + data.make_2aa_comb(vocab)
    
    # Load training and test data
    raw_training_data = data.load_csv(train_path)
    raw_test_data = data.load_csv(test_path)
    
    dataset_train_size = len(raw_training_data)
    dataset_test_size = len(raw_test_data)
    
    # Predefined parameters according to your sequence size
    prot_max_len = PROT_LENGTH 
    batch_size = BATCH_SIZE
    max_prot_mask = MAX_PROT_MASK
    
    # Parameters for the BERT
    num_layers = 3
    num_heads = 6
    
    prot_config = SimpleNamespace(
        vocab_size=len(VOCAB_2MER),
        hidden_size=120,
        max_position_embeddings=prot_max_len,
        type_vocab_size=1,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        num_attention_heads=num_heads,
        hidden_act="gelu",
        intermediate_size=160,
        num_hidden_layers=num_layers,
        is_decoder=False,
        output_attentions=True,
        output_hidden_states=True,
        pruned_heads = {},
        initializer_range=0.02,
        device=device
    )
    
    # Make data adequate for auxiliary task
    tokenizer = data.Tokenizer(k=kmer, vocab=VOCAB_2MER)
    _, train_dataset, test_dataset = data.create_datasets(raw_training_data, raw_test_data, tokenizer, prot_max_len, mask_length, max_prot_mask, task="auxiliary")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model and other parameters related to training
    model = models.BertForMaskedLM(config=prot_config, positional_embedding=models.PositionalEmbedding, attention=models.BertSelfAttention).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Train loop
    for epoch in range(150):
        total_train_loss = 0.0
        
        # Set model to train mode
        model.train()
        
        # Iterate over batches in train_loader
        for batch_input_ids, batch_segment_ids, batch_masked_lm_labels, _, _, batch_attention_mask in train_loader:
            optimizer.zero_grad()
            
            loss, outputs, hidden_states, _ = model(
                input_ids=batch_input_ids.to(device),
                token_type_ids=batch_segment_ids.to(device),
                masked_lm_labels=batch_masked_lm_labels.to(device),
                attention_mask=batch_attention_mask.to(device))

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Every ten epochs evaluate the model
        if (epoch + 1) % 10 == 0:
            model.eval()
            total_eval_loss = 0.0
                
            for batch_input_ids, batch_segment_ids, batch_masked_lm_labels, _, _, batch_attention_mask in test_loader:
                with torch.no_grad():
                    loss, outputs, hidden_states, _ = model(
                        input_ids=batch_input_ids.to(device),
                        token_type_ids=batch_segment_ids.to(device),
                        masked_lm_labels=batch_masked_lm_labels.to(device),
                        attention_mask=batch_attention_mask.to(device))
        
                    if batch_attention_mask.sum() - torch.numel(batch_attention_mask) > 0 :
                        print("found patting", batch_attention_mask.sum())
                        
                    total_eval_loss += loss.item()
                    
            avg_eval_loss = total_eval_loss / len(test_loader)
            print('Epoch:', '%04d' % (epoch + 1), 'train cost =', '{:.6f}'.format(avg_train_loss), 'eval cost =', '{:.6f}'.format(avg_eval_loss))

    average_train_acc, _ = evaluation.model_masked_label_accuracy(model, train_loader, device)
    average_test_acc, last_test_attention = evaluation.model_masked_label_accuracy(model, test_loader, device)
    print('Train Acc =', '{:.6f}'.format(average_train_acc), 'Eval Acc =', '{:.6f}'.format(average_test_acc))
    
    return model

def train_main_loop(train_path=TRAIN_PATH, test_path=TEST_PATH, off_sequences_path=OFF_SEQUENCES_PATH, on_sequences_path=ON_SEQUENCES_PATH, balancing=True):
    """
    Function for training (cross-validation and hyperparameter tuning) for classification task.
    
    Args:
        train_path (str): Path to the .csv file for training sequences (for auxiliary task). Defaults to TRAIN_PATH.
        test_path (str): Path to the .csv file for test sequences (for auxiliary task). Defaults to TEST_PATH.
        off_sequences_path (str): Path to the .txt file for the sequences belonging to the OFF cluster. Defaults to OFF_SEQUENCES_PATH.
        on_sequences_path (str): Path to the .txt file for the sequences belonging to the ON cluster. Defaults to ON_SEQUENCES_PATH.
        balancing (bool): Whether we should balance the sequences or not (only for all sequences). Defaults to True.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First get trained model from auxiliary task
    model = train_auxiliary_task(train_path, test_path)

    # Freeze BERT parameters
    for params in model.parameters():
        params.requires_grad = False
    
    # Variables 
    vocab = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "S", "P", "T", "W", "Y", "V", "-"]
    VOCAB_2MER = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + data.make_2aa_comb(vocab)
    kmer = 2
    mask_length = kmer
    prot_max_len = PROT_LENGTH 
    max_prot_mask = MAX_PROT_MASK
    tokenizer = data.Tokenizer(k=kmer, vocab=VOCAB_2MER)
    
    if balancing == True:
        batch_size = BATCH_SIZE_BALANCED
    else:
        batch_size = BATCH_SIZE_CONSENSUS
    
    # Define all the variables necessary for classification
    raw_training_data, raw_test_data = data.balance_data(off_sequences_path, on_sequences_path, balance=balancing)
    dataset, train_dataset, test_dataset = data.create_datasets(raw_training_data, raw_test_data, tokenizer, prot_max_len, mask_length, max_prot_mask, task="classifier")
    
    # Define parameters for cross-validation
    num_epochs = 250
    k_folds = 5
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    criterion = nn.BCELoss()
    
    best_layer_structure = []
    best_lr = -1.0
    best_wd = -1.0
    best_model_acc = -1
    
    # Grid search parameters
    layer_structure = [[512, 256, 128, 64, 32], [128, 64, 1], [64, 32, 1]]
    learning_rate = [0.0001, 0.01, 0.00001]
    weight_decay = [0, 0.4, 0.04, 0.004]
    
    for layers in layer_structure:
        for lr in learning_rate:
            for wd in weight_decay:
                avg_test_acc = 0.0
                # Iterating over folds
                for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
                    
                    # Generate train and test splits for this fold
                    train_samples = SubsetRandomSampler(train_idx)
                    test_samples = SubsetRandomSampler(val_idx)
                    
                    # Create DataLoaders for these samples
                    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_samples)
                    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_samples)
                    
                    classifier = models.SequenceClassifier(model).to(device)
                    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
                    
                    for epoch in range(num_epochs):
                        # Train 
                        train_loss = helpers.train_epoch(classifier, device, train_loader, criterion, optimizer)
                        test_loss = helpers.valid_epoch(classifier, device, test_loader, criterion)
                    
                    # Take this accuracy as final accuracy of a fold
                    avg_test_acc += evaluation.classification_accuracy(classifier, test_loader, device)
                
                test_acc_results = avg_test_acc / k_folds
                print(test_acc_results)
                if test_acc_results > best_model_acc:
                    best_model_acc = test_acc_results
                    best_layer_structure = layers
                    best_lr = lr
                    best_wd = wd

    print("Best model hyperparameters:")           
    print("Best classifier architecture:", best_layer_structure)
    print("Best learning rate:", best_lr)
    print("Best weight decay:", best_wd)
                    

if __name__ == '__main__':
    train_main_loop()
    