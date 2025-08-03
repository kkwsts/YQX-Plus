#!/usr/bin/env python3
"""
Simplified End-to-End Regression Training Script for Aria.
Uses local paths and simplified configuration.

Training parameters aligned with the original paper:
- Learning rate: 1e-5 (without warmup)
- Schedule: Linear decay to 0
- Epochs: 10 epochs
- Dropout: Progressive residual dropout 0.0 -> 0.2
- Model: Uses end-of-sequence token for prediction
"""

import torch
import torch._dynamo
import os
import sys
import argparse
import logging
import accelerate
import json

# Add parent directory to path to import aria modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aria.config import load_model_config
from aria.utils import _load_weight, denormalize_score_tensor
from ariautils.tokenizer import AbsTokenizer
from aria.model import TransformerREG, ModelConfig
from aria.datasets import (
    RegressionDataset,
)

from torch import nn
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from logging.handlers import RotatingFileHandler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

LEARNING_RATE = 1e-5


def setup_logger(project_dir: str):
    """Setup logging for the training process."""
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    fh = RotatingFileHandler(
        os.path.join(project_dir, "logs.txt"), backupCount=5, maxBytes=1024**3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return get_logger(__name__)


def get_dataloaders(
    dataset_type: str = "all",
    batch_size: int = 8,
    num_workers: int = 4,
    max_seq_len: int = 1024,
):
    """Create train and validation dataloaders."""
    tokenizer = AbsTokenizer()

    if dataset_type == "ycu":
        train_dataset = RegressionDataset(
            midi_dir="./dataset/YCU-PPE-III-Midi/midi",
            csv_file="./dataset/YCU-PPE-III-Midi/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="train",
        )
        val_dataset = RegressionDataset(
            midi_dir="./dataset/YCU-PPE-III-Midi/midi",
            csv_file="./dataset/YCU-PPE-III-Midi/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="test",
        )
    elif dataset_type == "augmented":
        train_dataset = RegressionDataset(
            midi_dir="./dataset/augmented_performances/midi",
            csv_file="./dataset/augmented_performances/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="train",
        )
        val_dataset = RegressionDataset(
            midi_dir="./dataset/augmented_performances/midi",
            csv_file="./dataset/augmented_performances/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="test",
        )

    elif dataset_type == "all":
        # Create a custom combined dataset with all three datasets
        from torch.utils.data import ConcatDataset

        # YCU datasets
        ycu_train = RegressionDataset(
            midi_dir="./dataset/YCU-PPE-III-Midi/midi",
            csv_file="./dataset/YCU-PPE-III-Midi/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="train",
        )
        ycu_val = RegressionDataset(
            midi_dir="./dataset/YCU-PPE-III-Midi/midi",
            csv_file="./dataset/YCU-PPE-III-Midi/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="test",
        )


        # Augmented datasets
        aug_train = RegressionDataset(
            midi_dir="./dataset/augmented_performances/midi",
            csv_file="./dataset/augmented_performances/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="train",
        )
        aug_val = RegressionDataset(
            midi_dir="./dataset/augmented_performances/midi",
            csv_file="./dataset/augmented_performances/train_test_split.csv",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split="test",
        )

        train_dataset = ConcatDataset([ycu_train, aug_train])
        val_dataset = ConcatDataset([ycu_val, aug_val])

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def get_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    scheduler_type: str = "linear",
):
    """Setup optimizer and scheduler following paper specifications."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = num_epochs * steps_per_epoch

    if scheduler_type == "linear":
        # Linear decay as specified in paper (without warmup)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,  # Decay to 0 as in paper
            total_iters=total_steps,
        )
    elif scheduler_type == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_steps,
            power=0.5,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.1
        )
    elif scheduler_type == "constant":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return optimizer, scheduler


def train_model(
    dataset_type: str = "combined",
    num_epochs: int = 10,
    batch_size: int = 8,
    scheduler_type: str = "linear",
    checkpoint_path: str = "./ckpt/base.safetensors",
    max_seq_len: int = 1024,
    project_dir: str = "./experiments",
):
    """Main training function."""
    accelerator = accelerate.Accelerator(
        project_dir=project_dir,
        gradient_accumulation_steps=1,
    )

    if accelerator.is_main_process:
        os.makedirs(project_dir, exist_ok=True)
        logger = setup_logger(project_dir)
    else:
        logger = get_logger(__name__)

    logger.info(f"Training dataset: {dataset_type}")
    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    logger.info(f"Learning rate: {LEARNING_RATE} (paper-aligned)")
    logger.info(f"Scheduler: {scheduler_type} decay to 0")
    logger.info(f"Max sequence length: {max_seq_len}")
    logger.info(f"Progressive dropout: 0.0 -> 0.2 (residual connections)")

    # Setup tokenizer and model
    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config("medium-regression"))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.max_seq_len = max_seq_len
    model = TransformerREG(model_config)

    # Disable compilation for Windows compatibility
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    torch.set_float32_matmul_precision("high")

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model_state = _load_weight(checkpoint_path)
        model_state = {
            k.replace("_orig_mod.", ""): v for k, v in model_state.items()
        }
        model.load_state_dict(model_state, strict=False)
    else:
        logger.info("Training from scratch (no checkpoint found)")

    # Setup data loaders
    train_dataloader, val_dataloader = get_dataloaders(
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=4,
        max_seq_len=max_seq_len,
    )

    # Setup optimizer and scheduler
    optimizer, scheduler = get_optim(
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        scheduler_type=scheduler_type,
    )

    # Prepare for distributed training
    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )

    # Training loop
    loss_fn = nn.MSELoss()
    best_r2 = -float("inf")
    epoch_metrics = []

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"
    )

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Training Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process,
        )

        for sequences, targets in pbar:
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = loss_fn(predictions, targets)
            accelerator.backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_steps

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for sequences, targets in tqdm(
                val_dataloader,
                desc=f"Validation Epoch {epoch+1}",
                disable=not accelerator.is_local_main_process,
            ):
                predictions = model(sequences)
                loss = loss_fn(predictions, targets)

                val_loss += loss.item()
                val_steps += 1

                # Denormalize for metrics
                pred_denorm = (
                    denormalize_score_tensor(predictions).cpu().numpy()
                )
                targ_denorm = denormalize_score_tensor(targets).cpu().numpy()

                predictions_list.extend(pred_denorm.flatten())
                targets_list.extend(targ_denorm.flatten())

        val_loss /= val_steps
        mae = mean_absolute_error(targets_list, predictions_list)
        r2 = r2_score(targets_list, predictions_list)

        # Log results
        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, MAE: {mae:.2f}, R²: {r2:.4f}"
        )

        # Save metrics
        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": mae,
                "val_r2": r2,
            }
        )

        # Save best model
        if r2 > best_r2:
            best_r2 = r2
            if accelerator.is_main_process:
                model_path = os.path.join(project_dir, "best_model.pt")
                torch.save(
                    accelerator.unwrap_model(model).state_dict(), model_path
                )
                logger.info(f"New best R²: {best_r2:.4f} - saved model")

    # Save final results
    if accelerator.is_main_process:
        results = {
            "dataset_type": dataset_type,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "scheduler_type": scheduler_type,
            "best_r2": best_r2,
            "epoch_metrics": epoch_metrics,
        }

        with open(os.path.join(project_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training completed. Best R²: {best_r2:.4f}")

    return best_r2


def main():
    parser = argparse.ArgumentParser(
        description="Train end-to-end regression model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["ycu", "augmented", "all"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "poly", "cosine", "constant"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ckpt/base.safetensors",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="./experiments",
        help="Project directory for outputs",
    )

    args = parser.parse_args()

    print("=== End-to-End Regression Training ===")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Project dir: {args.project_dir}")

    train_model(
        dataset_type=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        checkpoint_path=args.checkpoint,
        max_seq_len=args.max_seq_len,
        project_dir=args.project_dir,
    )


if __name__ == "__main__":
    main()
