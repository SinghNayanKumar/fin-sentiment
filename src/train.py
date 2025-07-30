import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from collections import defaultdict
import pandas as pd
import config

def train_epoch(model, data_loader, optimizer, device, scheduler):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # The model only accepts arguments it's designed for.
        model_inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }
        labels = batch["labels"].to(device).unsqueeze(1)
        
        # Pass model_inputs using **kwargs and labels separately
        outputs = model(**model_inputs, labels=labels)

        
        loss = outputs.loss  # This is MSELoss by default for regression
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'MSE_loss': loss.item()})
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def score_to_label_eval(score):
    """Converts a predicted score to a label based on thresholds in config."""
    if score > config.POSITIVE_THRESHOLD:
        return 2  # Positive
    elif score < config.NEGATIVE_THRESHOLD:
        return 0  # Negative
    else:
        return 1  # Neutral

def evaluate(model, data_loader, device):
    """
    Evaluates the regression model and provides both overall and per-type 
    classification metrics.
    """
    model.eval()
    
    all_predicted_scores = []
    all_true_scores = []
    all_type_ids = []  # Store document type IDs for stratified analysis
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_scores = batch["labels"]
            type_ids = batch["type_ids"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Squeeze to remove the last dimension: [batch_size, 1] -> [batch_size]
            predicted_scores = outputs.logits.squeeze(-1)
            
            all_predicted_scores.extend(predicted_scores.cpu().numpy())
            all_true_scores.extend(true_scores.cpu().numpy())
            all_type_ids.extend(type_ids.cpu().numpy())

    # --- 1. Overall Metrics ---
    overall_mse = mean_squared_error(all_true_scores, all_predicted_scores)
    overall_mae = mean_absolute_error(all_true_scores, all_predicted_scores)
    
    # Perform post-hoc classification on all data
    predicted_labels = [score_to_label_eval(s) for s in all_predicted_scores]
    true_labels = [score_to_label_eval(s) for s in all_true_scores]
    
    overall_class_report = classification_report(
        true_labels, predicted_labels,
        target_names=["Negative", "Neutral", "Positive"], output_dict=True, zero_division=0
    )

    # --- 2. Per-Type Metrics ---
    per_type_reports = {}
    preds_by_type = defaultdict(list)
    labels_by_type = defaultdict(list)

    # Group predictions and labels by their document type
    for i, type_id in enumerate(all_type_ids):
        preds_by_type[type_id].append(predicted_labels[i])
        labels_by_type[type_id].append(true_labels[i])
    
    # Generate a separate classification report for each document type
    for type_id, type_name in config.ID2TYPE.items():
        if type_id in preds_by_type:  # Check if this type exists in the current data split
            report = classification_report(
                labels_by_type[type_id],
                preds_by_type[type_id],
                target_names=["Negative", "Neutral", "Positive"], output_dict=True, zero_division=0
            )
            per_type_reports[type_name] = report
            
    # --- 3. Return a comprehensive dictionary of results ---
    return {
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "overall_report": overall_class_report,
        "per_type_reports": per_type_reports
    }