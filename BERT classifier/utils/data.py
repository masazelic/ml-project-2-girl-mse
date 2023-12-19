# .py file for Tokenizer and masking data
import random
import re
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

class Tokenizer(object):
    """ Class for tokenizing a sequence. """

    def __init__(self, k, vocab, unknown="[UNK]"):
        """
        Initialization of class object.
        
        k (int): Lenght of one token (k-mer).
        vocab (list): List containing all the vocabulary.
        """
        self.k = k
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(vocab)}
        self.unknown = unknown
        if self.unknown not in self.token2idx:
            self.token2idx[self.unknown] = len(self.token2idx)

    def tokenize(self, text):
        """ Turn sequence into tokens according to _parse_text() function. """
        return [token if token in self.token2idx else "[UNK]" for token in self._parse_text(text)]

    def convert_tokens_to_ids(self, tokens):
        """ Turns tokens into ids according to self.token2idx mapping. """
        return [self.token2idx.get(token, self.token2idx[self.unknown]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """ Turns ids to tokens. Reverse form of the previous function. """
        return [self.idx2token.get(token_id, self.unknown) for token_id in ids]
    
    def _parse_text(self, text):
        """ Parse sequence in k-mers. """
        output = []
        for i in range(len(text) - (self.k-1)):
          output.append(text[i:i+self.k])
        
        output.insert(0, '[CLS]')
        output.append('[SEP]')
        return output
    
    def random_token_id(self):
        """ Take random id of token. """
        return random.randint(0, len(self.idx2token))
    
def load_csv(filepath):
    return pd.read_csv(filepath)

def make_2aa_comb(vocab):
  """ 
  Makes 2-mers vocabulary for amino acid sequences.

  Args:
  vocab (list): all of the letters corresponding to 20 amino acids and dash in MSA (multiple sequence alignments

  Return:
  aa2_comb (list): vocabulary of all 2 letter combinations

  """
  
  aa2_comb = []
  for aa1 in vocab:
    for aa2 in vocab:
        aa2_single = ""
        aa2_single = aa1 + aa2
        aa2_comb.append(aa2_single)

  return aa2_comb

def generate_labeled_data(data, tokenizer, max_len, max_size=None):
    """ 
    Returns token ids and class labels for data.

    Args:
    data (pd.DataFrame): Data (sequences and labels) to be tokenized.
    tokenizer (Tokenizer): Used for tokenization.
    max_len (int): Max lenght of sequence.
    max_size (int): Number of samples from data to return.

    """
    if max_size is not None:
        data = data[:max_size]
    labels = data['label'].values  # Extracts numpy array from DataFrame
    return _generate_masked_data(data, tokenizer, max_len, k=0, mask_rate=0.0, max_mask=0, noise_rate=0.0, max_size=max_size)[:1] + (torch.tensor(labels),)

def generate_masked_data(data, tokenizer, max_len, k, mask_rate=0.3, max_mask=3, noise_rate=0.0, max_size=None):
    """ 
    Generating masked data for auxiliary task.
    Description of the parameters in _generate_masked_data() function.
    """
    return _generate_masked_data(data, tokenizer, max_len, k, mask_rate, max_mask, noise_rate, max_size)

def _generate_masked_data(data, tokenizer, max_len, k, mask_rate, max_mask, noise_rate, max_size):
    """
    Generate masked data for masked language model (MLM) training.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'sequence' column with text sequences.
        tokenizer: Tokenizer.
        max_len (int): Maximum length of the generated sequences.
        k (int): The number of continuous positions to mask in a single masking operation.
        mask_rate (float): Percentage of tokens to mask in each sequence.
        max_mask (int): Maximum number of tokens to mask in a single sequence.
        noise_rate (float): Percentage of tokens to randomly replace with noise tokens.
        max_size (int or None): Maximum number of rows to use from the input data.

    Returns:
        Tuple of torch tensors containing:
        - input_ids: Tokenized and masked input sequences.
        - segment_ids: Segment IDs (0 for single-segment sequences).
        - all_masked_lm_labels: Masked language model labels for prediction.
        - all_label_idxs: Indices of masked positions in the input sequences.
        - all_labels: Original labels for masked positions.
        - attention_masks: Attention masks for zero paddings.

    """
    default_ignore_label = -100
    # Limit the size of the input data
    if max_size is not None:
        data = data[:max_size]

    input_ids = []
    segment_ids = []
    all_masked_lm_labels = []
    all_labels = []
    all_label_idxs = []

    attention_masks = []
    for _, row in data.iterrows():
        sequence = row['sequence']

        # Tokenize the sequence
        tokens = tokenizer.tokenize(sequence)
        ids = tokenizer.convert_tokens_to_ids(tokens)

        # Randomly insert noise tokens
        masked_seq = ids.copy()
        n_to_noise = int(len(tokens) * noise_rate)
        to_noise = random.sample(range(len(tokens)), n_to_noise)
        for pos in to_noise:
                masked_seq[pos] = tokenizer.random_token_id()
        
        # Mask mask_rate of tokens
        n_to_mask = max(min(max_mask, int(len(tokens) * mask_rate)), 1)
        to_mask = random.sample(range(len(tokens)), n_to_mask)
        masked_seq = ids.copy()
        masked_lm_labels =  [default_ignore_label] * max_len

        for pos in to_mask:
            for continuous_pos in range(k):
                next_masked_pos = pos + continuous_pos
                if next_masked_pos >= len(tokens):
                    next_masked_pos = pos - continuous_pos
                masked_seq[next_masked_pos] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
                masked_lm_labels[next_masked_pos] = ids[pos]

        # Zero Paddings
        attention_mask = [1] * len(masked_seq)
        if max_len > len(masked_seq):
            n_pad = max_len - len(masked_seq)
            masked_seq.extend([0] * n_pad)
            attention_mask.extend([0] * n_pad)

        # label Paddings
        labels = [ids[i] for i in to_mask]
        if max_mask > len(to_mask):
            n_pad = max_mask - len(to_mask)
            labels.extend([default_ignore_label] * n_pad)
            to_mask.extend([default_ignore_label] * n_pad)


        input_ids.append(masked_seq)
        segment_ids.append([0] * len(masked_seq)) # single-segment sequences
        all_masked_lm_labels.append(masked_lm_labels)
        all_labels.append(labels)
        all_label_idxs.append(to_mask)
        attention_masks.append(attention_mask)
    
    return torch.tensor(input_ids), torch.tensor(segment_ids), torch.tensor(all_masked_lm_labels), torch.tensor(all_label_idxs), torch.tensor(all_labels), torch.tensor(attention_masks)

def balance_data(off_sequences_path, on_sequences_path, seed=42, ratio=0.3, balance=False):
  """
  Function for balancing unbalanced ON and OFF sequences.

  Args:
    off_sequences_path (str): Path to the .txt file that stores OFF sequences.
    on_sequences_path (str): Path to the .txt file that strores ON sequences.
  
  Returns:
    raw_training_data (pd.DataFrame): All training sequences.
    raw_test_data (pd.DataFrame): All test sequences.
  """
  # Set fixed seed 
  np.random.seed(seed)

  # Lists that are going to store off and on sequences
  data_off = []
  data_on = []

  with open(off_sequences_path) as file:
    for item in file:
      data_off.append(item)

  with open(on_sequences_path) as file:
    for item in file:
      data_on.append(item)
  
  # We know there is more OFF sequences
  length_on = len(data_on)
  length_off = len(data_off)
  data_off_balanced = []

  # If balance parameter is True, do the balancing
  # If not just merge
  if balance == True:
    select_indexes = np.random.choice(length_off, length_on, replace=False).tolist()

    # List for storing selected OFF sequences for balancing
    for i in select_indexes:
      data_off_balanced.append(data_off[i])
  
  else:
    data_off_balanced = data_off
  
  # Merge data on and data off
  seq_for_cls = data_off_balanced + data_on

  # Create labels for these sequences
  labels_off = np.zeros((len(data_off_balanced)), dtype=int)
  labels_on = np.ones((len(data_on)), dtype=int)
  data_labels = np.concatenate((labels_off, labels_on))
  data_labels = data_labels.tolist()

  # Turn to pd
  dict = {'sequence' : seq_for_cls, 'label': data_labels}
  data_frame = pd.DataFrame(dict)
  data_frame = data_frame.sample(frac = 1).reset_index(drop=True)

  # Split data on train and test 
  train_data, test_data = train_test_split(data_frame, test_size=ratio, random_state=seed, stratify=data_frame.label)
  
  # Create data appropriate for the model
  raw_training_data = train_data.reset_index(drop=True)
  raw_test_data = test_data.reset_index(drop=True)

  return raw_training_data, raw_test_data

def create_datasets(raw_training_data, raw_test_data, tokenizer, dna_max_len, mask_length, max_dna_mask, task="classifier"):
  """
  Creates Data Loaders for specified task.

  Args:
    raw_training_data (pd.DataFrame): Contains all training sequences.
    raw_test_data (pd.DataFrame): Contains all test sequences.
    tokenizer (Tokenizer): Tokenizer used to tokenize the sequence.
    dna_max_len (int): Maximum lenght of generated sequence.
    mask_length (int): The number of continuous positions to mask in a single masking operation.
    max_dna_max (int): Maximum number of tokens to mask in a single sequence.
    task (str): "classifier" or "auxiliary"; Specifies task for which we are creating training/test datasets.
  
  Returns:
    dataset (torch.Dataset): Concatenated train_dataset and test_dataset.
    train_dataset (torch.Dataset): Train dataset.
    test_dataset (torch.Dataset): Test dataset.
  """

  if task == "classifier":

    # Obtaine input_ids and class_labels for classification task
    input_ids, class_labels = generate_labeled_data(raw_training_data, tokenizer, max_len=dna_max_len, max_size=len(raw_training_data))
    test_input_ids, test_class_labels = generate_labeled_data(raw_test_data, tokenizer, max_len=dna_max_len, max_size=len(raw_test_data))

    train_dataset = TensorDataset(input_ids, class_labels)
    test_dataset = TensorDataset(test_input_ids, test_class_labels)
  
  else:

    # Obtain elements for auxiliary task
    input_ids, segment_ids, masked_lm_labels, labels_idx, labels, attention_masks = generate_masked_data(raw_training_data, tokenizer, max_len=dna_max_len, max_mask=max_dna_mask, k=mask_length, mask_rate=0.05, max_size=len(raw_training_data))
    test_input_ids, test_segment_ids, test_masked_lm_labels, test_labels_idx, test_labels, test_attention_masks = generate_masked_data(raw_test_data, tokenizer, max_len=dna_max_len, max_mask=max_dna_mask, k=mask_length, mask_rate=0.05, max_size=len(raw_test_data))

    train_dataset = TensorDataset(input_ids, segment_ids, masked_lm_labels, labels_idx, labels, attention_masks)
    test_dataset = TensorDataset(test_input_ids, test_segment_ids, test_masked_lm_labels, test_labels_idx, test_labels, test_attention_masks)

  # Merge is important for cross-validation
  dataset = ConcatDataset([train_dataset, test_dataset])
  
  return dataset, train_dataset, test_dataset

