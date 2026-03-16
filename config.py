"""
Configuration file for TaxoClass (Hierarchical Multi-Label Text Classification).
Manages hyperparameters, file paths, and reproducibility settings.
"""

import random
import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional but recommended for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    """
    Configuration class for TaxoClass project.
    Contains all hyperparameters, file paths, and training settings.
    """
    
    # ==================== Reproducibility ====================
    SEED: int = 42
    
    # ==================== File Paths ====================
    # Base directory for data files
    BASE_DIR: str = "Amazon_products"
    
    # Training and test data
    TRAIN_FILE: str = "Amazon_products/train/train_corpus.txt"
    TEST_FILE: str = "Amazon_products/test/test_corpus.txt"
    
    # Taxonomy files
    TAXONOMY_FILE: str = "Amazon_products/class_hierarchy.txt"
    CLASSES_FILE: str = "Amazon_products/classes.txt"
    KEYWORDS_FILE: str = "Amazon_products/class_related_keywords.txt"
    # outputs/refined_keywords.json
    
    # Output directory
    OUTPUT_DIR: str = "outputs"
    EMBEDDINGS_DIR: str = "outputs/embeddings"
    LABELS_PATH: str = "outputs/labels.json"

    # ==================== Hardware ====================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==================== Model Settings ====================
    BERT_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    MAX_LENGTH: int = 128  # Tokenization length
    HIDDEN_DIM: int = 768  # Model hidden dimension (embedding size)
    VIRTUAL_ROOT_ID: int = 531

    # ==================== Generator Settings ====================
    # Generator model settings
    GENERATOR_SBERT_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    GENERATOR_LAMBDA_WEIGHT: float = 0.4  # Weight for parent embedding in propagation
    GENERATOR_CONFLICT_THRESHOLD: float = 0.05  # Threshold for conflict resolution
    GENERATOR_EMBEDDING_BATCH_SIZE: int = 32  # Batch size for embedding generation
    
    # Generator anchor set settings
    GENERATOR_MAX_SAMPLES_PER_PAIR: int = 10  # Max samples per domain pair for anchor set
    
    # Generator API settings
    GENERATOR_API_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    GENERATOR_MODEL_NAME: str = "meta-llama/llama-3.1-70b-instruct"
    GENERATOR_API_TEMPERATURE: float = 0.1
    GENERATOR_API_MAX_TOKENS: int = 512
    
    # ==================== Training Script Settings ====================
    # Training paths
    TRAIN_SAVE_DIR: str = "checkpoints"
    
    # Training hyperparameters
    TRAIN_BATCH_SIZE: int = 64 
    TRAIN_EPOCHS: int = 5
    TRAIN_LR: float = 5e-5
    TRAIN_WEIGHT_DECAY: float = 5e-6
    TRAIN_EPS: float = 1e-8
    TRAIN_GRAD_CLIP: float = 1.0
    TRAIN_GPU: int = 0

    # ==================== Model Settings (Optimizer) ====================
    MODEL_WEIGHT_DECAY: float = 5e-6
    MODEL_EPS: float = 1e-8
    MODEL_GRAD_CLIP: float = 1.0
    
    # ==================== Prediction Script Settings ====================
    # Prediction paths
    PREDICT_MODEL_PATH: str = "checkpoints/model.pt"
    PREDICT_OUTPUT_PATH: str = "2021320046_submission.csv"
    
    # Prediction hyperparameters
    PREDICT_BATCH_SIZE: int = 64
    PREDICT_MIN_LABELS: int = 2
    PREDICT_MAX_LABELS: int = 3
    PREDICT_OUTPUT_FORMAT: str = "csv"  # "csv" or "json"
    PREDICT_GPU: int = 0
    
    # ==================== Utility Methods ====================
    
    def ensure_output_dir(self) -> None:
        """Create output directory and cache directory if they don't exist."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def __post_init__(self):
        """Post-initialization: ensure output directory exists and fix paths."""
        self.ensure_output_dir()
        # Convert relative paths to absolute paths if needed
        # This ensures paths work correctly regardless of where the script is run from
        if not os.path.isabs(self.OUTPUT_DIR):
            self.OUTPUT_DIR = os.path.abspath(self.OUTPUT_DIR)
        if not os.path.isabs(self.LABELS_PATH):
            self.LABELS_PATH = os.path.abspath(self.LABELS_PATH)


# Create a global config instance
config = Config()

# Set random seed for reproducibility
seed_everything(config.SEED)

