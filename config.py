import torch

# -- Model and Tokenizer Configuration --
# Comparing 'bert-base-cased' and 'yiyang-zhang/finbert-tone'
#MODEL_NAME = 'yiyang-zhang/finbert-tone'
MODEL_NAME = 'bert-base-cased'

# -- Task Configuration --
# Switch between 'sentence' and 'absa' (Aspect-Based Sentiment Analysis)
TASK_TYPE = 'absa' 

# -- Data Configuration --
# These are the column names in your Hugging Face dataset
TEXT_COLUMN = 'sentence'
ASPECT_COLUMN = 'aspect'
LABEL_COLUMN = 'score'

# -- Training Hyperparameters --
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

# -- Label Mapping -- Treating as a regression problem to predict sentiment score
NUM_LABELS = 1

# -- Thresholds for Post-Hoc Classification --
# These will be used ONLY during evaluation, not training.
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1

# -- Environment --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42