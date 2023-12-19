# Code written by Marija Zelic, Sara Zatezalo and Elena Mrdja
# For the purpose of training and testing classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

def train_epoch(classifier, device, dataloader, criterion, optimizer):
  """
  Trains the model for one epoch.

  Args:
    classifier (nn.Module): Model we are training.
    device ("cpu" or "cuda"): Device at which we are doing training.
    dataloader (torch.DataLoader): Dataset data-loader.
    criterion (callable): Loss function use for training.
    optimizer (torch.optim): Optimizer used for training.

  Returns:
    train_loss (float): Train loss over all batches.

  """
  # Set model into train mode
  train_loss = 0.0
  for input_ids, class_labels in dataloader:
    input_ids = input_ids.to(device)
    class_labels = class_labels.float().unsqueeze(dim=1).to(device)

    # Reset gradients of all tracked variables
    optimizer.zero_grad()
    logits = classifier(input_ids)
    loss = criterion(logits, class_labels)

    # Calculate the loss for model outputs
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

  return train_loss / len(dataloader)

def valid_epoch(classifier, device, dataloader, criterion):
  """
  Evaluate one epoch of trained model.

  Args:
    model (nn.Module): Model we are training.
    device ("cpu" or "cuda"): Device at which we are doing training.
    dataloader (torch.DataLoader): Dataset data-loader.
    criterion (callable): Loss function use for training.

  Returns:
    valid_loss (float): Validation loss over all batches.
  """

  # Set the model to evaluation mode
  valid_loss = 0.0
  classifier.eval()

  for input_ids, class_labels in dataloader:
    with torch.no_grad():

      input_ids = input_ids.to(device)
      class_labels = class_labels.float().unsqueeze(dim=1).to(device)

      # Make predictions
      logits = classifier(input_ids)
      loss = criterion(logits, class_labels)
      valid_loss += loss.item()

  return valid_loss / len(dataloader)


