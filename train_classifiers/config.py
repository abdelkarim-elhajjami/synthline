from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model_name: str = 'bert-base-uncased'
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 6
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.06
    max_seq_length: int = 128

@dataclass
class PathConfig:
    """Path configuration."""
    input_dir: str = 'output'
    datasets_dir: str = 'train_classifiers/datasets'
    classifiers_dir: str = 'train_classifiers/classifiers'
    evaluations_dir: str = 'train_classifiers/evaluations' 