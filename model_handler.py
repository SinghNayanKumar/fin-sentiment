from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

def build_tokenizer():
    """Builds and returns a tokenizer from the specified model name in config."""
    print(f"Loading tokenizer for model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return tokenizer

def build_model():
    """
    Builds and returns a sequence classification model configured for REGRESSION.
    
    Setting num_labels=1 tells the Hugging Face model to:
    1.  Add a single output neuron (a regression head).
    2.  Use Mean Squared Error (MSELoss) as the default loss function during training.
    """
    print(f"Building regression model: {config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS  
    )
    return model