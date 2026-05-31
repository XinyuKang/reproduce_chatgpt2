from dataclasses import dataclass


@dataclass
class GPTConfig:
    # GPT-2 tokenizer vocab size
    vocab_size: int = 50257

    # Model size
    block_size: int = 1025
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.1

    # Training
    batch_size: int = 4
    grad_accum_steps: int = 16
    max_steps: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    num_epochs: int = 1

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    # System
    device: str = "cuda"
    dtype: str = "float16"

    # Paths
    raw_data_path: str = "data/raw/input.txt"
    train_data_path: str = "data/processed/train.bin"
    val_data_path: str = "data/processed/val.bin"
    checkpoint_dir: str = "checkpoints"