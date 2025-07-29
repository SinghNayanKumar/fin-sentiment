# data_loader.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import config

def create_data_loaders(tokenizer):
    """
    Loads the FIQA dataset from Hugging Face, preprocesses it for regression,
    and returns PyTorch DataLoaders.

    Args:
        tokenizer: An initialized tokenizer object (e.g., from AutoTokenizer).

    Returns:
        A tuple containing (train_dataloader, valid_dataloader, test_dataloader).
    """
    # 1. Load the dataset from Hugging Face Hub
    # The 'fiqa-sentiment-classification' dataset has 'train', 'validation', and 'test' splits.
    print("Loading dataset 'TheFinAI/fiqa-sentiment-classification' from Hugging Face Hub...")
    ds = load_dataset("TheFinAI/fiqa-sentiment-classification")


    print(f"Dataset loaded. Splits: {list(ds.keys())}")

    # Create a mapping from 'type' (an string) to an integer ID
    unique_types = ds['train'].unique('type')
    type2id = {type_name: i for i, type_name in enumerate(unique_types)}
    id2type = {i: type_name for type_name, i in type2id.items()}
    # We'll save this mapping to the config for use in evaluation
    config.ID2TYPE = id2type
    print(f"Document types found and mapped: {type2id}")


    # 2. Define the preprocessing function
    def preprocess_function(examples):
        # Format input text based on the task type from config.py
        if config.TASK_TYPE == 'absa':
            # Adding Aspect in front of Sentence for Aspect based sentiment analysis
            texts = [
                f"{aspect} {tokenizer.sep_token} {sentence}"
                for aspect, sentence in zip(examples[config.ASPECT_COLUMN], examples[config.TEXT_COLUMN])
            ]
        else: # Default to 'sentence' sentiment analysis task
            texts = examples[config.TEXT_COLUMN]

        # Tokenize the texts. Padding is handled later by the data collator.
        tokenized_inputs = tokenizer(
            texts,
            max_length=config.MAX_LENGTH,
            truncation=True
        )

        # For regression, the 'labels' are the float scores from the dataset.
        tokenized_inputs["labels"] = examples[config.LABEL_COLUMN]

        # Add the type_id to our processed data
        tokenized_inputs["type_ids"] = [type2id[t] for t in examples['type']]
        
        return tokenized_inputs

    # 3. Apply the preprocessing function to all splits of the dataset
    print("Tokenizing and formatting dataset...")
    columns_to_remove = ['_id', 'sentence', 'target', 'aspect', 'type']
    tokenized_datasets = ds.map(
        preprocess_function,
        batched=True,
        # Remove original text columns after processing. Keep 'score' to be renamed.
        remove_columns=columns_to_remove
    )


    # The labels are already named 'score', so we rename to 'labels'.
    tokenized_datasets = tokenized_datasets.rename_column("score", "labels")
    tokenized_datasets.set_format("torch",columns=["input_ids", "attention_mask", "labels", "type_ids"])

    # 4. Create a Data Collator for dynamic padding
    # This pads each batch to the length of the longest sequence in that batch.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        collate_fn=data_collator
    )
    
    valid_dataloader = DataLoader(
        tokenized_datasets["valid"],
        batch_size=config.BATCH_SIZE,
        collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=config.BATCH_SIZE,
        collate_fn=data_collator
    )

    print("DataLoaders created successfully.")
    return train_dataloader, valid_dataloader, test_dataloader