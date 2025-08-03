"""Contains miscellaneous utilities"""

import torch


def _load_weight(ckpt_path: str, device="cpu"):
    if ckpt_path.endswith("safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                f"Please install safetensors in order to read from the checkpoint: {ckpt_path}"
            ) from e
        return load_file(ckpt_path, device=device)
    else:
        import torch

        return torch.load(ckpt_path, map_location=device)


# Score normalization utilities for regression
def normalize_score(score: float) -> float:
    """Normalize score from 0-100 scale to 0-1 scale."""
    return score / 100.0


def denormalize_score(normalized_score: float) -> float:
    """Convert normalized score (0-1) back to 0-100 scale."""
    return normalized_score * 100.0


def normalize_score_tensor(scores: torch.Tensor) -> torch.Tensor:
    """Normalize score tensor from 0-100 scale to 0-1 scale."""
    return scores / 100.0


def denormalize_score_tensor(normalized_scores: torch.Tensor) -> torch.Tensor:
    """Convert normalized score tensor (0-1) back to 0-100 scale."""
    return normalized_scores * 100.0
