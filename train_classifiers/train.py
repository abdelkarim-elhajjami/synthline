"""Training script."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, List
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as transformers_logging,
    get_linear_schedule_with_warmup
    
)
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
from data_processor import DataProcessor
from config import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore', message='Some weights of BertForSequenceClassification')

def prepare_data(texts, labels, tokenizer, max_length=128):
    """Prepare data for model training."""
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(labels.tolist(), dtype=torch.long)
    )
    return dataset

def _get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _train_model(model, train_dataloader, device, num_epochs, optimizer, scheduler):
    """Train the model and return training statistics."""
    model.train()
    training_stats = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        training_stats.append({
            'epoch': epoch + 1,
            'avg_loss': avg_epoch_loss
        })
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
    
    return training_stats

def _save_classifier(model, tokenizer, output_dir: Path, classifier_name: str, training_stats: List[Dict]):
    """Save classifier artifacts."""
    save_path = output_dir / classifier_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    stats_path = save_path / 'training_stats.txt'
    with open(stats_path, 'w') as f:
        for epoch_stat in training_stats:
            f.write(f"Epoch {epoch_stat['epoch']}: Loss = {epoch_stat['avg_loss']:.4f}\n")
    
    logger.info(f"Classifier and training stats saved to {save_path}")

def train_and_save_classifier(classifier_name: str, train_data: pd.DataFrame, config: TrainingConfig, classifiers_dir: Path):
    """Train and save a classifier."""
    device = _get_device()
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(DataProcessor.LABEL_MAP),
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Prepare training data
    train_dataset = prepare_data(
        train_data['requirement'], 
        train_data['class'], 
        tokenizer,
        config.max_seq_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset, generator=torch.Generator()),
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # Train and save classifier
    training_stats = _train_model(
        model, train_dataloader, device,
        config.num_epochs, optimizer, scheduler
    )
    _save_classifier(model, tokenizer, classifiers_dir, classifier_name, training_stats)

def main():

    device = _get_device()
    logger.info(f"Using device: {device}")
    
    # Load configurations
    config = TrainingConfig()
    datasets_dir = Path("train_classifiers/datasets")
    classifiers_dir = Path("train_classifiers/classifiers")
    classifiers_dir.mkdir(parents=True, exist_ok=True)
    
    # Define classifiers to train
    classifiers_to_train = {
        'deepseek': 'deepseek.csv',
        'gpt4o': 'gpt4o.csv',
        'gpt4o_deepseek': 'gpt4o_deepseek.csv',
        'real_train': 'real_train.csv',
        'real_train_deepseek': 'real_train_deepseek.csv',
        'real_train_gpt4o': 'real_train_gpt4o.csv',
        'real_train_gpt4o_deepseek': 'real_train_gpt4o_deepseek.csv'
    }
    
    # Train all classifiers
    for classifier_name, dataset_file in classifiers_to_train.items():
        logger.info(f"\nTraining {classifier_name}...")
        train_data = pd.read_csv(datasets_dir / dataset_file, sep=';')
        train_and_save_classifier(classifier_name, train_data, config, classifiers_dir)

if __name__ == "__main__":
    main() 