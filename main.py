# main.py
import torch
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import copy

import config
import utils
import data_loader
import model_handler
import training

# Initialize logger
logger = utils.get_logger(__name__)

def run():
    """
    Main function to orchestrate the training and evaluation pipeline.
    """
    # For reproducibility
    utils.set_seed(config.RANDOM_SEED)

    logger.info("--- Starting Financial Sentiment Analysis Pipeline ---")
    logger.info(f"Configuration: Model={config.MODEL_NAME}, Task={config.TASK_TYPE}, Device={config.DEVICE}")

    # --- 1. Build Tokenizer and Model ---
    tokenizer = model_handler.build_tokenizer()
    model = model_handler.build_model()
    model.to(config.DEVICE)

    # --- 2. Load and Prepare Data ---
    # The data_loader handles downloading, preprocessing, and creating DataLoaders
    # It also populates config.ID2TYPE with the document type mapping.
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(tokenizer)

    # --- 3. Setup Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # --- 4. Training Loop ---
    best_val_f1 = 0
    best_model_state = None

    logger.info(f"--- Starting Training for {config.NUM_EPOCHS} Epochs ---")
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        avg_train_loss = training.train_epoch(model, train_loader, optimizer, config.DEVICE, scheduler)
        logger.info(f"Average Training MSE Loss: {avg_train_loss:.4f}")
        
        # Evaluate on the validation set
        val_results = training.evaluate(model, val_loader, config.DEVICE)
        val_f1 = val_results['overall_report']['weighted avg']['f1-score']
        val_mse = val_results['overall_mse']
        
        logger.info(f"Validation MSE: {val_mse:.4f} | Validation Weighted F1: {val_f1:.4f}")
        
        # Save the best model based on validation F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"New best model found with F1: {best_val_f1:.4f} and saved.")

    # --- 5. Final Evaluation on Test Set ---
    logger.info("--- Training Finished. Starting Final Evaluation on Test Set ---")
    # Load the best performing model for the final test
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("No best model was saved. Evaluating the model from the last epoch.")
    
    test_results = training.evaluate(model, test_loader, config.DEVICE)
    
    # Display results in a clean format
    logger.info("--- Overall Test Results ---")
    logger.info(f"Regression MSE: {test_results['overall_mse']:.4f}")
    logger.info(f"Regression MAE: {test_results['overall_mae']:.4f}")
    
    overall_df = pd.DataFrame(test_results['overall_report']).transpose()
    logger.info("Overall Classification Report:\n" + overall_df.to_string())

    logger.info("\n--- Per-Type Test Results ---")
    for type_name, report in test_results['per_type_reports'].items():
        logger.info(f"\n--- Report for Document Type: '{type_name}' ---")
        type_df = pd.DataFrame(report).transpose()
        logger.info("\n" + type_df.to_string())

    logger.info("--- Pipeline Finished ---")

if __name__ == '__main__':
    run()