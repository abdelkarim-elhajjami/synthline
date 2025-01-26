"""Script for evaluating diversity metrics of generated text datasets."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _get_device() -> torch.device:
    """Get the best available compute device."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def _generate_ngrams(text: str, n: int = 3) -> List[Tuple[str, ...]]:
    """Generate n-grams from cleaned text."""
    cleaned = text.replace('.', '').replace('\n', ' ').strip()
    words = [w for w in cleaned.split() if w]
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def _calculate_ingf(texts: List[str], n: int = 3) -> float:
    """Calculate Inter-sample N-gram Frequency."""
    all_ngrams = [ngram for text in texts for ngram in _generate_ngrams(text, n)]
    if not all_ngrams:
        return 0.0
    
    unique_ngrams = set(all_ngrams)
    return sum(all_ngrams.count(ngram) for ngram in unique_ngrams) / len(unique_ngrams)

def _calculate_vocabulary_size(texts: List[str]) -> int:
    """Calculate unique vocabulary size."""
    return len({word for text in texts for word in text.split()})

def _calculate_cosine_similarities(
    texts: List[str], 
    labels: List[int], 
    device: torch.device, 
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate intra-class and inter-class cosine similarities."""
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens').to(device)
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[i:i+batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**batch).last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(outputs)
    
    embeddings = np.vstack(embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    labels = np.array(labels)
    
    # Create similarity masks
    intra_mask = (labels[:, None] == labels) & ~np.eye(len(labels), dtype=bool)
    inter_mask = labels[:, None] != labels
    
    return similarity_matrix[intra_mask], similarity_matrix[inter_mask]

def calculate_metrics(df: pd.DataFrame, device: torch.device) -> Dict:
    """Calculate comprehensive diversity metrics."""
    texts = df['requirement'].tolist()
    labels = df['class'].astype(int).tolist()
    
    intra_sim, inter_sim = _calculate_cosine_similarities(texts, labels, device)
    
    return {
        'vocabulary_size': _calculate_vocabulary_size(texts),
        'average_pairwise_similarity': float(np.concatenate([intra_sim, inter_sim]).mean()),
        'intra_class_aps': float(intra_sim.mean()),
        'inter_class_aps': float(inter_sim.mean()),
        'inter_ngram_frequency': _calculate_ingf(texts),
        'intra_similarities': intra_sim.tolist(),
        'inter_similarities': inter_sim.tolist()
    }

def _plot_distribution(data: List[float], color: str, label: str) -> None:
    """Helper function to plot a single distribution."""
    density = gaussian_kde(data)
    xs = np.linspace(-0.25, 1.0, 200)
    plt.plot(xs, density(xs), color=color, label=label, linewidth=2)
    plt.fill_between(xs, density(xs), color=color, alpha=0.2)

def plot_similarity_distribution(results: Dict, output_dir: Path) -> None:
    """Visualize intra-class similarity distributions."""
    plt.figure(figsize=(10, 6))
    
    color_mapping = {
        'real_train': '#1f77b4',
        'deepseek': '#ff7f0e',
        'gpt4o': '#2ca02c'
    }
    
    for dataset_name, metrics in results.items():
        try:
            _plot_distribution(
                metrics['intra_similarities'],
                color=color_mapping.get(dataset_name, '#d62728'),
                label=dataset_name
            )
        except Exception as e:
            logger.warning(f"Skipping {dataset_name}: {str(e)}")
    
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Estimated Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / 'intra_class_similarity_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_diversity_results(results: Dict, output_dir: Path) -> None:
    """Save metrics and generate visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical metrics
    metrics = {
        name: {k: v for k, v in data.items() if not k.endswith('_similarities')}
        for name, data in results.items()
    }
    
    with (output_dir / f'diversity_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    plot_similarity_distribution(results, output_dir)

def main() -> None:
    """Main execution workflow."""
    
    device = _get_device()
    logger.info(f"Initialized compute device: {device}")
    
    config = PathConfig()
    data_dir = Path('train_classifiers/datasets')
    
    try:
        datasets = {
            'real_train': pd.read_csv(data_dir / 'real_train.csv', sep=';'),
            'deepseek': pd.read_csv(data_dir / 'deepseek.csv', sep=';'),
            'gpt4o': pd.read_csv(data_dir / 'gpt4o.csv', sep=';')
        }
        
        results = {}
        for name, df in datasets.items():
            logger.info(f"Processing dataset: {name}")
            results[name] = calculate_metrics(df, device)
        
        eval_dir = Path(config.evaluations_dir) / 'diversity_evaluation'
        save_diversity_results(results, eval_dir)
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()