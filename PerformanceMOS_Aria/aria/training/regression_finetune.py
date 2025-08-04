"""Training script for regression fine-tuning on YCU-PPE dataset."""

import torch
import torch._dynamo
import os
import argparse
import logging
import accelerate
import json

from aria.config import load_model_config
from aria.utils import _load_weight, denormalize_score_tensor
from ariautils.tokenizer import AbsTokenizer
from aria.model import TransformerREG, ModelConfig
from aria.datasets import RegressionDataset, CombinedRegressionDataset

from torch import nn
from torch.utils.data import DataLoader

from accelerate.logging import get_logger
from logging.handlers import RotatingFileHandler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

LEARNING_RATE = 1e-4


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


def setup_project_dir(project_dir: str | None = None):
    """Setup project directory for saving results."""
    if project_dir is None:
        project_dir = "./experiments"

    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "checkpoints"), exist_ok=True)

    return project_dir


def get_dataloaders(
    ycu_midi_dir: str,
    ycu_csv_file: str,
    asap_midi_dir: str,
    asap_csv_file: str,
    batch_size: int,
    num_workers: int,
    max_seq_len: int = 1024,
):
    """Create train and validation dataloaders for combined YCU and ASAP datasets."""
    tokenizer = AbsTokenizer()

    train_dataset = CombinedRegressionDataset(
        ycu_midi_dir=ycu_midi_dir,
        ycu_csv_file=ycu_csv_file,
        asap_midi_dir=asap_midi_dir,
        asap_csv_file=asap_csv_file,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="train",
    )
    val_dataset = CombinedRegressionDataset(
        ycu_midi_dir=ycu_midi_dir,
        ycu_csv_file=ycu_csv_file,
        asap_midi_dir=asap_midi_dir,
        asap_csv_file=asap_csv_file,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="test",
    )

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
    scheduler_type: str = "cosine",
    warmup_epochs: int = 0,
    min_lr_factor: float = 0.1,
):
    """Setup optimizer and scheduler with different scheduler options."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_type == "cosine":
        # Standard cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=LEARNING_RATE * min_lr_factor
        )
    elif scheduler_type == "cosine_warmup":
        # Cosine with warmup - starts low, increases, then decreases
        from torch.optim.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if warmup_steps > 0 else 0.1,
            anneal_strategy="cos",
        )
    elif scheduler_type == "linear":
        # Linear decay - more gradual than cosine
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_factor,
            total_iters=total_steps,
        )
    elif scheduler_type == "step":
        # Step decay - drops learning rate at specific epochs
        step_size = max(1, num_epochs // 4)  # Drop every 1/4 of training
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.5
        )
    elif scheduler_type == "plateau":
        # Reduce LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
    elif scheduler_type == "poly":
        # Polynomial decay - gentle start, controlled end
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_steps,
            power=0.5,  # Square root decay - gentler than linear
        )
    elif scheduler_type == "constant":
        # Constant learning rate
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return optimizer, scheduler


def _train(
    num_epochs: int,
    accelerator: accelerate.Accelerator,
    model: TransformerREG,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    project_dir: str | None = None,
):
    """Main training loop."""
    logger = get_logger(__name__)
    loss_fn = nn.MSELoss()

    # def make_checkpoint(
    #     _accelerator: accelerate.Accelerator, _epoch: int, _step: int
    # ):
    #     if accelerator.is_main_process:
    #         checkpoint_dir = os.path.join(
    #             project_dir,
    #             "checkpoints",
    #             f"epoch{_epoch}_step{_step}",
    #         )
    #         logger.info(
    #             f"EPOCH {_epoch}/{num_epochs}: Saving checkpoint - {checkpoint_dir}"
    #         )
    #         _accelerator.save_state(checkpoint_dir)
    def save_checkpoint(
        _accelerator: accelerate.Accelerator,
        _epoch: int,
        _step: int,
        checkpoint_type: str = "latest",
    ):
        """Save checkpoint - either 'best' or 'latest'."""
        if accelerator.is_main_process:
            model_path = os.path.join(
                project_dir, f"{checkpoint_type}_model.pt"
            )
            logger.info(
                f"EPOCH {_epoch}/{num_epochs}: Saving {checkpoint_type} model - {model_path}"
            )
            torch.save(accelerator.unwrap_model(model).state_dict(), model_path)

    def train_loop(dataloader: DataLoader, _epoch: int):
        model.train()
        total_loss = 0
        loss_buffer = []

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Training Epoch {_epoch}",
            disable=not accelerator.is_local_main_process,
        )

        try:
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
        except Exception:
            lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

        for step, (sequences, targets) in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                predictions = model(sequences)  # (batch_size, 1)
                loss = loss_fn(predictions, targets)

                accelerator.backward(loss)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
                avg_loss = sum(loss_buffer) / len(loss_buffer)

                pbar.set_postfix_str(
                    f"lr={lr_for_print}, loss={loss.item():.4f}, avg_loss={avg_loss:.4f}"
                )

        return sum(loss_buffer) / len(loss_buffer)

    def val_loop(dataloader: DataLoader, _epoch: int):
        model.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for sequences, targets in tqdm(
                dataloader,
                desc=f"Validation Epoch {_epoch}",
                disable=not accelerator.is_local_main_process,
            ):
                predictions = model(sequences)
                loss = loss_fn(predictions, targets)

                total_loss += loss.item()

                # Denormalize for metrics calculation
                pred_denorm = (
                    denormalize_score_tensor(predictions).cpu().numpy()
                )
                targ_denorm = denormalize_score_tensor(targets).cpu().numpy()

                predictions_list.extend(pred_denorm.flatten())
                targets_list.extend(targ_denorm.flatten())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        mae = mean_absolute_error(targets_list, predictions_list)
        r2 = r2_score(targets_list, predictions_list)

        return avg_loss, mae, r2

    # Training loop with early stopping
    epoch_metrics = []
    best_r2 = -float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = 5

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Early stopping patience: {early_stopping_patience} epochs")
    logger.info(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"
    )

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        train_loss = train_loop(train_dataloader, epoch)
        val_loss, val_mae, val_r2 = val_loop(val_dataloader, epoch)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_r2": val_r2,
        }
        epoch_metrics.append(metrics)

        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val MAE: {val_mae:.2f}, "
            f"Val R²: {val_r2:.4f}"
        )

        # Check for best model (based on R²)
        if val_r2 > best_r2:
            best_r2 = val_r2
            epochs_without_improvement = 0
            save_checkpoint(accelerator, epoch, 0, "best")
            logger.info(f"New best R²: {best_r2:.4f} - saved best model")
        else:
            epochs_without_improvement += 1

        # Save latest model every epoch
        save_checkpoint(accelerator, epoch, 0, "latest")

        # Step plateau scheduler if using it
        if scheduler and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered after {early_stopping_patience} epochs "
                f"without improvement. Best R²: {best_r2:.4f}"
            )
            break

    logger.info(f"Training completed. Best R²: {best_r2:.4f}")
    return epoch_metrics


def train(
    model_name: str,
    ycu_midi_dir: str,
    ycu_csv_file: str,
    asap_midi_dir: str,
    asap_csv_file: str,
    num_workers: int,
    num_epochs: int,
    batch_size: int,
    grad_acc_steps: int,
    project_dir: str | None = None,
    checkpoint_path: str | None = None,
    max_seq_len: int = 1024,
    scheduler_type: str = "poly",
    warmup_epochs: int = 0,
    min_lr_factor: float = 0.1,
):
    """Main training function."""
    accelerator = accelerate.Accelerator(
        project_dir=project_dir,
        gradient_accumulation_steps=grad_acc_steps,
    )

    if accelerator.is_main_process:
        project_dir = setup_project_dir(project_dir)
        logger = setup_logger(project_dir)
    else:
        project_dir = project_dir or "./experiments"
        logger = get_logger(__name__)

    logger.info(f"Project directory: {project_dir}")
    logger.info(f"YCU MIDI directory: {ycu_midi_dir}")
    logger.info(f"YCU CSV file: {ycu_csv_file}")
    logger.info(f"ASAP MIDI directory: {asap_midi_dir}")
    logger.info(f"ASAP CSV file: {asap_csv_file}")
    logger.info(f"Max sequence length: {max_seq_len}")
    logger.info(
        f"Training config: epochs={num_epochs}, batch_size={batch_size}, "
        f"num_workers={num_workers}, grad_acc_steps={grad_acc_steps}"
    )
    logger.info(f"Scheduler type: {scheduler_type}")
    if warmup_epochs > 0:
        logger.info(f"Warmup epochs: {warmup_epochs}")
    logger.info(f"Min LR factor: {min_lr_factor}")

    # Setup tokenizer and model
    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.max_seq_len = max_seq_len
    model = TransformerREG(model_config)

    # Disable automatic compilation to avoid Triton issues on Windows
    torch._dynamo.config.suppress_errors = True

    # Set float32 matrix multiplication precision for better performance
    torch.set_float32_matmul_precision("high")

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model_state = _load_weight(checkpoint_path)
        model_state = {
            k.replace("_orig_mod.", ""): v for k, v in model_state.items()
        }
        model.load_state_dict(model_state, strict=False)
    else:
        logger.info("No checkpoint path provided, training from scratch")

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True

    # model.compile()

    # Setup data loaders
    train_dataloader, val_dataloader = get_dataloaders(
        ycu_midi_dir=ycu_midi_dir,
        ycu_csv_file=ycu_csv_file,
        asap_midi_dir=asap_midi_dir,
        asap_csv_file=asap_csv_file,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_len=max_seq_len,
    )

    # Setup optimizer and scheduler
    optimizer, scheduler = get_optim(
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        min_lr_factor=min_lr_factor,
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

    # Train the model
    epoch_metrics = _train(
        num_epochs=num_epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        project_dir=project_dir,
    )

    # Save final results
    best_r2 = (
        max(metric["val_r2"] for metric in epoch_metrics)
        if epoch_metrics
        else 0.0
    )
    best_mae = (
        min(metric["val_mae"] for metric in epoch_metrics)
        if epoch_metrics
        else float("inf")
    )

    logger.info(f"Best R²: {best_r2:.4f}")
    logger.info(f"Best MAE: {best_mae:.2f}")

    results = {
        "model_name": model_name,
        "ycu_midi_dir": ycu_midi_dir,
        "ycu_csv_file": ycu_csv_file,
        "asap_midi_dir": asap_midi_dir,
        "asap_csv_file": asap_csv_file,
        "max_seq_len": max_seq_len,
        "scheduler_type": scheduler_type,
        "epoch_metrics": epoch_metrics,
        "best_r2": best_r2,
        "best_mae": best_mae,
    }

    if accelerator.is_main_process:
        with open(os.path.join(project_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train regression model for score prediction on combined YCU-PPE and ASAP datasets."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model configuration name"
    )
    parser.add_argument(
        "--ycu_midi_dir",
        type=str,
        required=True,
        help="Path to YCU-PPE-III-Midi directory",
    )
    parser.add_argument(
        "--ycu_csv_file",
        type=str,
        required=True,
        help="Path to YCU train_test_split.csv file",
    )
    parser.add_argument(
        "--asap_midi_dir",
        type=str,
        required=True,
        help="Path to ASAP MIDI directory",
    )
    parser.add_argument(
        "--asap_csv_file",
        type=str,
        required=True,
        help="Path to ASAP dataset_info_with_splits.csv file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=None,
        help="Project directory for saving results",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="poly",
        help="Type of learning rate scheduler (cosine, cosine_warmup, linear, step, plateau, poly, constant)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Number of epochs for warmup in cosine_warmup scheduler",
    )
    parser.add_argument(
        "--min_lr_factor",
        type=float,
        default=0.1,
        help="Factor to reduce learning rate by at the end of training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model_name,
        ycu_midi_dir=args.ycu_midi_dir,
        ycu_csv_file=args.ycu_csv_file,
        asap_midi_dir=args.asap_midi_dir,
        asap_csv_file=args.asap_csv_file,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        project_dir=args.project_dir,
        checkpoint_path=args.checkpoint_path,
        max_seq_len=args.max_seq_len,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
    )
