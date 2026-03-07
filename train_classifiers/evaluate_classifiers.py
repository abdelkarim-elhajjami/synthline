"""Evaluation script for trained classifiers."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from config import PathConfig
from train import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _evaluate_model(model, eval_dataloader, device):
    """Evaluate model and return detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'macro_metrics': {
            'precision': precision,
            'recall': recall
        },
        'confusion_matrix': conf_matrix.tolist()
    }

def _save_evaluation_results(results: dict, output_dir: Path, model_name: str):
    """Save evaluation results with descriptive filenames."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results with model name in filename
    results_file = output_dir / f'results_{model_name}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_file}")

def main():
    """Main function to evaluate all classifiers."""
    
    device = _get_device()
    logger.info(f"Using device: {device}")
    
    # Load configurations
    paths = PathConfig()
    project_root = Path(__file__).parent.parent
    
    # Load test dataset
    real_test = pd.read_csv(project_root / paths.datasets_dir / 'real_test.csv', sep=';')
    real_test['class'] = real_test['class'].astype(int)
    
    # Classifiers to evaluate
    classifiers = [
        'deepseek',
        'gpt4o',
        'gpt4o_deepseek',
        'real_train',
        'real_train_deepseek',
        'real_train_gpt4o',
        'real_train_gpt4o_deepseek'
    ]
    
    # Create classifiers_evaluation directory
    evaluations_dir = project_root / 'train_classifiers/evaluations/classifiers_evaluation'
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate all classifiers on test set
    logger.info("\nEvaluating on test set:")
    for classifier_name in classifiers:
        logger.info(f"\nEvaluating {classifier_name}...")
        
        classifier_path = project_root / 'train_classifiers/classifiers' / classifier_name
        model = AutoModelForSequenceClassification.from_pretrained(classifier_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        
        test_dataset = prepare_data(real_test['requirement'], real_test['class'], tokenizer)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            sampler=SequentialSampler(test_dataset)
        )
        
        results = _evaluate_model(model, test_dataloader, device)
        _save_evaluation_results(results, evaluations_dir, classifier_name)
        
        logger.info(f"Precision: {results['macro_metrics']['precision']:.4f}")
        logger.info(f"Recall: {results['macro_metrics']['recall']:.4f}")

if __name__ == "__main__":
    main()