# Code written by Sara Zatezalo, Marija Zelic and Elena Mrdja for the purpose of
# Evaluation of implemented models

import random
import torch

def masked_label_accuracy(labels, labels_idx, outputs):
    """
    Calculate masked label accuracy for a batch of predictions.

    Args:
        labels (torch.Tensor): Tensor containing original labels for masked positions.
        labels_idx (torch.Tensor): Tensor containing indices of masked positions in the input sequences.
        outputs (torch.Tensor): Model predictions.

    Returns:
        float: Masked label accuracy.

    """
    acc_masked = 0
    total_count = 0

    # Get the predicted tokens with the highest probability
    tokens_predictions = outputs.max(2)[1]
    for batch in range(len(labels)):

        # Filter out positions with negative indices (paddings)
        token_idxs_mask, token_labels_mask = labels_idx[batch] >= 0, labels[batch]  >= 0
        token_idxs, token_labels = labels_idx[batch][token_idxs_mask], labels[batch][token_labels_mask]

        # Extract predicted tokens for masked positions
        tokens_prediction = tokens_predictions[batch][token_idxs]

        # Count correctly predicted masked tokens
        acc_masked += sum(tokens_prediction == token_labels).item()
        total_count += len(token_labels)

    return acc_masked / total_count

def model_masked_label_accuracy(model, data_loader, device):
    """
    Calculate the average masked label accuracy for a model on a given data loader.

    Args:
        model: Pre-trained language model.
        data_loader (DataLoader): PyTorch DataLoader containing batches of masked language model data.
        device (str): Device ('cuda' or 'cpu') on which the model and data should be loaded.

    Returns:
        Tuple[float, torch.Tensor]: Average masked label accuracy and attentions from the last batch.

    """
    with torch.no_grad():

        acc_masked = 0

        for batch_input_ids, batch_segment_ids, batch_masked_lm_labels, batch_masked_pos, batch_masked_tokens, _ in data_loader:
            
            # Forward pass through the model
            _, outputs, _, attentions = model(
            input_ids=batch_input_ids.to(device),
            token_type_ids=batch_segment_ids.to(device),
            masked_lm_labels=batch_masked_lm_labels.to(device),
            )
            
            # Calculate masked label accuracy for the batch
            local_acc = masked_label_accuracy(batch_masked_tokens, batch_masked_pos, outputs.data.detach().to("cpu"))
            acc_masked += local_acc
        
        # Calculate average masked label accuracy
        average_acc = acc_masked / len(data_loader)

    return average_acc, attentions

# Function for calculating the accuracy
def classification_accuracy(model, data_loader, device):
  """
  Calculate classification accuracy.

  Args: 
    model (nn.Module): Model which predictions we are evaluating.
    data_loader (torch.utils.DataLoader): Data Loader for the data we are evaluating.
    device (torch.device): "cpu" or "cuda"
  
  Returns:
    accuracy (float): Model prediction accuracy.
  """
  correct_predictions = 0
  total_count = 0
  for input_ids, labels in data_loader:
      with torch.no_grad():
          input_ids = input_ids.to(device)
          labels = labels.float().unsqueeze(dim=1).to(device)

          logits = model(input_ids)

          preds = (logits > 0.5).float()
          correct_predictions += (preds == labels).sum().item()
          total_count += len(labels)

  accuracy = correct_predictions / total_count

  return accuracy