#!/usr/bin/env python3
"""
Unified evaluation script for aria regression model.
Evaluates performance on Vienna, YCU, and ASAP datasets and generates separate CSV files.
"""

import os
import torch
import pandas as pd
import numpy as np
import logging
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import re

# Import aria modules - adjust paths for eval subdirectory
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from aria.model import TransformerREG, ModelConfig
from aria.config import load_model_config
from aria.utils import _load_weight, denormalize_score_tensor
from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def setup_logging(log_filename: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename),
        ],
    )
    return logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    model_config: dict,
    tokenizer: AbsTokenizer,
    device: str = "cuda",
) -> TransformerREG:
    """Load the regression model from checkpoint."""
    logger = logging.getLogger(__name__)

    config = ModelConfig(
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        ff_mult=model_config["ff_mult"],
        drop_p=model_config["drop_p"],
        max_seq_len=model_config["max_seq_len"],
        grad_checkpoint=model_config["grad_checkpoint"],
        resid_dropout=model_config.get("resid_dropout", 0.0),
        class_size=1,
    )

    config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerREG(config)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = _load_weight(checkpoint_path, device=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


def preprocess_midi(
    midi_path: str, tokenizer: AbsTokenizer, max_seq_len: int = 1024
) -> torch.Tensor:
    """Load and preprocess MIDI file for inference."""
    logger = logging.getLogger(__name__)

    try:
        midi_dict = MidiDict.from_midi(midi_path)
        tokenized_seq = tokenizer.tokenize(midi_dict)

        seq = tokenized_seq[:max_seq_len]
        if tokenizer.eos_tok not in seq:
            seq[-1] = tokenizer.eos_tok

        seq_tensor = torch.tensor(tokenizer.encode(seq))
        return seq_tensor

    except Exception as e:
        logger.error(f"Error processing {midi_path}: {e}")
        raise


def predict_score(
    model: TransformerREG, input_tensor: torch.Tensor, device: str = "cuda"
) -> float:
    """Predict score for a given input tensor."""
    model.eval()

    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)

        # Denormalize the score
        score = denormalize_score_tensor(output.squeeze(0))

        return float(score.item())


# =============================================================================
# VIENNA DATASET EVALUATION
# =============================================================================


def extract_song_name_five_songs(filename: str) -> str:
    """Extract song name treating 5 songs as separate entities."""
    base_name = filename.replace(".mid", "")
    song_name = re.sub(r"_p\d+$", "", base_name)

    if "Chopin_op10_no3" in song_name:
        return "Chopin_op10_no3"
    elif "Chopin_op38_1st-3rd" in song_name:
        return "Chopin_op38_1st-3rd"
    elif "Chopin_op38" in song_name and "1st-3rd" not in song_name:
        return "Chopin_op38"
    elif "Mozart_K331_1st-mov" in song_name:
        return "Mozart_K331_1st-mov"
    elif "Schubert_D783_no15" in song_name:
        return "Schubert_D783_no15"
    else:
        return song_name


def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics for a list of scores."""
    if not scores:
        return {}

    scores_array = np.array(scores)
    return {
        "mean": float(np.mean(scores_array)),
        "median": float(np.median(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "std": float(np.std(scores_array)),
        "variance": float(np.var(scores_array)),
        "count": len(scores),
    }


def process_vienna_folder(
    folder_path: str,
    model: TransformerREG,
    tokenizer: AbsTokenizer,
    device: str,
    max_seq_len: int,
) -> List[Tuple[str, float]]:
    """Process all MIDI files in a folder for Vienna dataset."""
    logger = logging.getLogger(__name__)
    results = []

    midi_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".mid")]
    )

    for midi_file in midi_files:
        full_path = os.path.join(folder_path, midi_file)
        filename = os.path.basename(midi_file)
        logger.info(f"Processing {filename}")

        input_tensor = preprocess_midi(full_path, tokenizer, max_seq_len)
        predicted_score = predict_score(model, input_tensor, device)

        results.append((filename, predicted_score))
        logger.info(f"{filename}: {predicted_score:.2f}")

    return results


def create_vienna_results_table(
    human_results: List[Tuple[str, float]],
    score_results: List[Tuple[str, float]],
    predicted_results: List[Tuple[str, float]],
    model_info: str,
) -> pd.DataFrame:
    """Create Vienna results table."""
    songs = [
        "Chopin_op10_no3",
        "Chopin_op38_1st-3rd",
        "Chopin_op38",
        "Mozart_K331_1st-mov",
        "Schubert_D783_no15",
    ]

    human_by_song = defaultdict(list)
    for filename, score in human_results:
        if "average" in filename.lower():
            continue
        song_name = extract_song_name_five_songs(filename)
        human_by_song[song_name].append(score)

    score_lookup = {
        extract_song_name_five_songs(filename): score
        for filename, score in score_results
    }
    predicted_lookup = {
        extract_song_name_five_songs(filename): score
        for filename, score in predicted_results
    }

    table_data = []
    for song in songs:
        if song in human_by_song:
            human_scores = human_by_song[song]
            human_stats = calculate_statistics(human_scores)

            predicted_score = predicted_lookup.get(song, None)
            predicted_str = (
                f"{predicted_score:.2f}"
                if predicted_score is not None
                else "N/A"
            )

            table_data.append(
                {
                    "Song": song,
                    "Score_Method": score_lookup.get(song, "N/A"),
                    "Predicted_Method": predicted_str,
                    "Human_Mean": f"{human_stats.get('mean', 0):.2f}",
                    "Human_Median": f"{human_stats.get('median', 0):.2f}",
                    "Human_Min": f"{human_stats.get('min', 0):.2f}",
                    "Human_Max": f"{human_stats.get('max', 0):.2f}",
                }
            )

    # Add global human statistics
    all_human_scores = [
        score for scores in human_by_song.values() for score in scores
    ]
    if all_human_scores:
        global_stats = calculate_statistics(all_human_scores)

        table_data.append(
            {
                "Song": "",
                "Score_Method": "",
                "Predicted_Method": "",
                "Human_Mean": "",
                "Human_Median": "",
                "Human_Min": "",
                "Human_Max": "",
            }
        )
        table_data.append(
            {
                "Song": "=== GLOBAL HUMAN STATISTICS ===",
                "Score_Method": "",
                "Predicted_Method": "",
                "Human_Mean": "",
                "Human_Median": "",
                "Human_Min": "",
                "Human_Max": "",
            }
        )
        table_data.append(
            {
                "Song": "Global_Statistics",
                "Score_Method": "",
                "Predicted_Method": "",
                "Human_Mean": f"{global_stats['mean']:.2f}",
                "Human_Median": f"{global_stats['median']:.2f}",
                "Human_Min": f"{global_stats['min']:.2f}",
                "Human_Max": f"{global_stats['max']:.2f}",
            }
        )

    return pd.DataFrame(table_data)


def evaluate_vienna_dataset(
    model: TransformerREG,
    tokenizer: AbsTokenizer,
    device: str,
    max_seq_len: int,
    model_info: str,
) -> str:
    """Evaluate Vienna dataset and save results."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Vienna dataset evaluation")

    test_dir = Path("./vienna-test")
    subfolders = ["human", "score", "predicted"]
    all_results = {}

    for subfolder in subfolders:
        folder_path = test_dir / subfolder
        if not folder_path.exists():
            logger.warning(f"Folder {folder_path} does not exist, skipping")
            all_results[subfolder] = []
            continue

        logger.info(f"Processing Vienna folder: {subfolder}")
        results = process_vienna_folder(
            str(folder_path), model, tokenizer, device, max_seq_len
        )
        all_results[subfolder] = results

    # Create comprehensive results table
    comprehensive_table = create_vienna_results_table(
        all_results.get("human", []),
        all_results.get("score", []),
        all_results.get("predicted", []),
        model_info,
    )

    # Save results in eval folder
    output_filename = f"vienna_scores_{model_info}.csv"
    comprehensive_table.to_csv(output_filename, index=False)

    logger.info(f"Vienna results saved to {output_filename}")
    return output_filename


# =============================================================================
# YCU DATASET EVALUATION
# =============================================================================


class PerformanceAnalyzer:
    """Comprehensive performance analysis for YCU dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_comprehensive_analysis(self) -> Dict:
        """Run all analysis modules."""
        return {
            "basic_metrics": self.calculate_basic_metrics(),
            "evaluator_agreement": self.analyze_evaluator_agreement(),
            "performance_buckets": self.analyze_performance_buckets(),
            "ranking_accuracy": self.analyze_ranking_accuracy(),
            "confidence_calibration": self.analyze_confidence_calibration(),
        }

    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic regression metrics."""
        true_scores = self.df["score_average"]
        pred_scores = self.df["predicted_score"]

        mae = np.mean(np.abs(pred_scores - true_scores))
        rmse = np.sqrt(np.mean((pred_scores - true_scores) ** 2))

        pearson_corr, _ = stats.pearsonr(true_scores, pred_scores)
        spearman_corr, _ = stats.spearmanr(true_scores, pred_scores)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_correlation": float(pearson_corr),
            "spearman_correlation": float(spearman_corr),
        }

    def analyze_evaluator_agreement(self) -> Dict:
        """Analyze agreement with individual evaluators."""
        results = {}
        evaluators = ["score_a", "score_b", "score_c"]

        best_corr = -1
        best_evaluator = None

        for evaluator in evaluators:
            corr, _ = stats.pearsonr(
                self.df[evaluator], self.df["predicted_score"]
            )
            results[f"{evaluator}_correlation"] = float(corr)

            if corr > best_corr:
                best_corr = corr
                best_evaluator = evaluator

        results["best_matching_evaluator"] = best_evaluator
        results["best_correlation"] = float(best_corr)

        # Human inter-rater reliability
        human_correlations = []
        for i, eval1 in enumerate(evaluators):
            for j, eval2 in enumerate(evaluators):
                if i < j:
                    corr, _ = stats.pearsonr(self.df[eval1], self.df[eval2])
                    human_correlations.append(corr)

        results["human_inter_rater_reliability"] = float(
            np.mean(human_correlations)
        )
        return results

    def analyze_performance_buckets(self) -> Dict:
        """Analyze performance by score ranges."""
        buckets = {
            "excellent": (90, 100),
            "good": (80, 89.99),
            "average": (70, 79.99),
            "poor": (60, 69.99),
            "very_poor": (0, 59.99),
        }

        results = {}
        for bucket_name, (min_score, max_score) in buckets.items():
            mask = (self.df["score_average"] >= min_score) & (
                self.df["score_average"] <= max_score
            )
            bucket_data = self.df[mask]

            if len(bucket_data) > 0:
                mae = np.mean(
                    np.abs(
                        bucket_data["predicted_score"]
                        - bucket_data["score_average"]
                    )
                )
                results[bucket_name] = {
                    "count": len(bucket_data),
                    "mae": float(mae),
                    "mean_bias": float(
                        np.mean(
                            bucket_data["predicted_score"]
                            - bucket_data["score_average"]
                        )
                    ),
                    "percentage": float(len(bucket_data) / len(self.df) * 100),
                }
            else:
                results[bucket_name] = {
                    "count": 0,
                    "mae": 0.0,
                    "mean_bias": 0.0,
                    "percentage": 0.0,
                }

        return results

    def analyze_ranking_accuracy(self) -> Dict:
        """Analyze pairwise ranking accuracy."""
        results = {}

        correct_pairs = 0
        total_pairs = 0

        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                true_i = self.df.iloc[i]["score_average"]
                true_j = self.df.iloc[j]["score_average"]
                pred_i = self.df.iloc[i]["predicted_score"]
                pred_j = self.df.iloc[j]["predicted_score"]

                if true_i != true_j:  # Skip ties
                    true_ranking = true_i > true_j
                    pred_ranking = pred_i > pred_j

                    if true_ranking == pred_ranking:
                        correct_pairs += 1
                    total_pairs += 1

        if total_pairs > 0:
            results["pairwise_ranking_accuracy"] = float(
                correct_pairs / total_pairs
            )
        else:
            results["pairwise_ranking_accuracy"] = 0.0

        results["total_pairs_compared"] = total_pairs
        return results

    def analyze_confidence_calibration(self) -> Dict:
        """Analyze model confidence vs accuracy."""
        results = {}

        # High confidence predictions (extreme scores)
        high_pred_mask = (self.df["predicted_score"] >= 85) | (
            self.df["predicted_score"] <= 65
        )
        high_conf_data = self.df[high_pred_mask]

        if len(high_conf_data) > 0:
            high_conf_mae = np.mean(
                np.abs(
                    high_conf_data["predicted_score"]
                    - high_conf_data["score_average"]
                )
            )
            results["high_confidence_mae"] = float(high_conf_mae)
            results["high_confidence_count"] = len(high_conf_data)

        # Conservative vs bold predictions
        pred_std = np.std(self.df["predicted_score"])
        true_std = np.std(self.df["score_average"])
        results["prediction_std"] = float(pred_std)
        results["true_std"] = float(true_std)
        results["conservatism_ratio"] = float(
            pred_std / true_std if true_std > 0 else 0
        )

        # Large disagreements
        large_errors = (
            np.abs(self.df["predicted_score"] - self.df["score_average"]) > 15
        )
        results["large_error_count"] = int(np.sum(large_errors))
        results["large_error_percentage"] = float(np.mean(large_errors) * 100)

        return results


def create_ycu_summary_report(analysis_results: Dict, model_info: str) -> str:
    """Create human-readable summary report for YCU results."""
    basic = analysis_results["basic_metrics"]
    evaluator = analysis_results["evaluator_agreement"]
    buckets = analysis_results["performance_buckets"]
    ranking = analysis_results["ranking_accuracy"]
    confidence = analysis_results["confidence_calibration"]

    report = f"""
YCU-PPE-III Dataset Evaluation Report
Model: {model_info}
=====================================

BASIC PERFORMANCE METRICS
--------------------------
Mean Absolute Error (MAE): {basic['mae']:.3f} points
Root Mean Square Error (RMSE): {basic['rmse']:.3f} points
Pearson Correlation: {basic['pearson_correlation']:.3f}
Spearman Correlation: {basic['spearman_correlation']:.3f}

EVALUATOR AGREEMENT ANALYSIS
----------------------------
Best matching evaluator: {evaluator['best_matching_evaluator']}
Best correlation: {evaluator['best_correlation']:.3f}
Human inter-rater reliability: {evaluator['human_inter_rater_reliability']:.3f}

PERFORMANCE BY SCORE RANGE
---------------------------
Excellent (90-100): {buckets['excellent']['count']} samples, MAE: {buckets['excellent']['mae']:.3f}
Good (80-89): {buckets['good']['count']} samples, MAE: {buckets['good']['mae']:.3f}
Average (70-79): {buckets['average']['count']} samples, MAE: {buckets['average']['mae']:.3f}
Poor (60-69): {buckets['poor']['count']} samples, MAE: {buckets['poor']['mae']:.3f}
Very Poor (0-59): {buckets['very_poor']['count']} samples, MAE: {buckets['very_poor']['mae']:.3f}

RANKING ACCURACY
----------------
Pairwise ranking accuracy: {ranking['pairwise_ranking_accuracy']:.3f}
Total pairs compared: {ranking['total_pairs_compared']}

CONFIDENCE CALIBRATION
----------------------
Large error percentage (>15 points): {confidence['large_error_percentage']:.1f}%
Prediction std: {confidence['prediction_std']:.2f}
True score std: {confidence['true_std']:.2f}
Conservatism ratio: {confidence['conservatism_ratio']:.2f}
"""
    return report


def evaluate_ycu_dataset(
    model: TransformerREG,
    tokenizer: AbsTokenizer,
    device: str,
    max_seq_len: int,
    model_info: str,
) -> str:
    """Evaluate YCU dataset and save results."""
    logger = logging.getLogger(__name__)
    logger.info("Starting YCU dataset evaluation")

    # Load test data
    data_df = pd.read_csv("./YCU-PPE-III-Midi/train_test_split.csv")
    test_data = data_df[data_df["split"] == "test"].copy()

    logger.info(f"Processing {len(test_data)} YCU test samples")

    # Run inference
    predictions = []
    midi_dir = Path("./YCU-PPE-III-Midi/midi")

    for idx, row in test_data.iterrows():
        filename = row["filename"]
        midi_path = midi_dir / f"{filename}.mid"

        if not midi_path.exists():
            logger.warning(f"MIDI file not found: {midi_path}")
            continue

        logger.info(f"Processing {filename}")

        input_tensor = preprocess_midi(str(midi_path), tokenizer, max_seq_len)
        predicted_score = predict_score(model, input_tensor, device)

        result = {
            "filename": filename,
            "folder_id": row["folder_id"],
            "score_a": row["score_a"],
            "score_b": row["score_b"],
            "score_c": row["score_c"],
            "score_average": row["score_average"],
            "predicted_score": predicted_score,
        }
        predictions.append(result)

        logger.info(
            f"{filename}: predicted={predicted_score:.2f}, true={row['score_average']:.2f}"
        )

    # Create results DataFrame
    results_df = pd.DataFrame(predictions)

    # Run comprehensive analysis
    analyzer = PerformanceAnalyzer(results_df)
    analysis_results = analyzer.run_comprehensive_analysis()

    # Save outputs in eval folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ycu_inference_{model_info}_{timestamp}"

    # 1. Detailed CSV
    detailed_csv = f"{base_filename}_detailed.csv"
    results_df.to_csv(detailed_csv, index=False)

    logger.info(f"YCU results saved: {detailed_csv}")
    return detailed_csv







# =============================================================================
# CUSTOM MIDI FOLDER EVALUATION
# =============================================================================


def evaluate_custom_midi_folder(
    model: TransformerREG,
    tokenizer: AbsTokenizer,
    device: str,
    max_seq_len: int,
    model_info: str,
    custom_midi_dir: str,
) -> str:
    """Evaluate MIDI files in a custom directory and save results."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting custom MIDI folder evaluation: {custom_midi_dir}")

    midi_dir = Path(custom_midi_dir)
    if not midi_dir.exists():
        logger.error(f"Custom MIDI directory not found: {custom_midi_dir}")
        return ""

    # Find all MIDI files
    midi_files = []
    for pattern in ["*.mid", "*.midi"]:
        midi_files.extend(list(midi_dir.glob(pattern)))

    if not midi_files:
        logger.error(f"No MIDI files found in {custom_midi_dir}")
        return ""

    logger.info(f"Found {len(midi_files)} MIDI files in {custom_midi_dir}")

    # Run inference
    predictions = []

    for midi_path in sorted(midi_files):
        filename = midi_path.name
        logger.info(f"Processing {filename}")

        try:
            input_tensor = preprocess_midi(
                str(midi_path), tokenizer, max_seq_len
            )
            predicted_score = predict_score(model, input_tensor, device)

            result = {
                "filename": filename,
                "file_path": str(midi_path),
                "predicted_score": predicted_score,
            }
            predictions.append(result)

            logger.info(f"{filename}: predicted={predicted_score:.2f}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            continue

    if not predictions:
        logger.error("No MIDI files were successfully processed")
        return ""

    # Create results DataFrame
    results_df = pd.DataFrame(predictions)

    # Calculate basic statistics
    scores = results_df["predicted_score"]
    stats_info = {
        "total_files": len(results_df),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "std_score": float(np.std(scores)),
    }

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = Path(custom_midi_dir).name
    detailed_csv = f"custom_{folder_name}_{model_info}_{timestamp}_detailed.csv"
    results_df.to_csv(detailed_csv, index=False)

    logger.info(f"Custom folder results saved: {detailed_csv}")
    return detailed_csv


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================


def extract_model_info_from_path(checkpoint_path: str) -> str:
    """Extract model information from checkpoint path."""
    path_obj = Path(checkpoint_path)

    # Try to extract epoch/step info
    match = re.search(r"epoch(\d+)_step(\d+)", str(path_obj))
    if match:
        epoch, step = match.groups()
        return f"epoch{epoch}_step{step}"

    # Try other patterns
    if "best_model" in str(path_obj):
        return "best_model"
    elif "latest_model" in str(path_obj):
        return "latest_model"
    else:
        return path_obj.stem


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for Vienna, YCU, and augmented datasets"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["vienna"],
        choices=["vienna", "ycu"],
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--custom_midi_dir",
        type=str,
        help="Path to folder containing MIDI files for inference (alternative to predefined datasets)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )

    args = parser.parse_args()

    # Setup logging in eval folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"unified_eval_{timestamp}.log"
    logger = setup_logging(log_filename)

    # Check if custom MIDI folder is provided
    if args.custom_midi_dir:
        logger.info("Starting custom MIDI folder evaluation")
        logger.info(f"Checkpoint: {args.checkpoint_path}")
        logger.info(f"Custom folder: {args.custom_midi_dir}")
    else:
        logger.info("Starting unified evaluation")
        logger.info(f"Checkpoint: {args.checkpoint_path}")
        logger.info(f"Datasets: {args.datasets}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint not found: {args.checkpoint_path}")
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        return

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_config = load_model_config("medium-regression")
    tokenizer = AbsTokenizer()
    model = load_model(args.checkpoint_path, model_config, tokenizer, device)

    model_info = extract_model_info_from_path(args.checkpoint_path)

    # Results summary
    results_files = []

    # Handle custom MIDI folder
    if args.custom_midi_dir:
        try:
            custom_file = evaluate_custom_midi_folder(
                model,
                tokenizer,
                device,
                args.max_seq_len,
                model_info,
                args.custom_midi_dir,
            )
            if custom_file:
                results_files.append(f"Custom folder: {custom_file}")
        except Exception as e:
            logger.error(f"Custom folder evaluation failed: {e}")
            print(f"Custom folder evaluation failed: {e}")
    else:
        # Evaluate predefined datasets
        if "vienna" in args.datasets:
            try:
                vienna_file = evaluate_vienna_dataset(
                    model, tokenizer, device, args.max_seq_len, model_info
                )
                results_files.append(f"Vienna: {vienna_file}")
            except Exception as e:
                logger.error(f"Vienna evaluation failed: {e}")
                print(f"Vienna evaluation failed: {e}")

        if "ycu" in args.datasets:
            try:
                ycu_file = evaluate_ycu_dataset(
                    model, tokenizer, device, args.max_seq_len, model_info
                )
                results_files.append(f"YCU: {ycu_file}")
            except Exception as e:
                logger.error(f"YCU evaluation failed: {e}")
                print(f"YCU evaluation failed: {e}")

        if "asap" in args.datasets:
            try:
                asap_file = evaluate_asap_dataset(
                    model, tokenizer, device, args.max_seq_len, model_info
                )
                if asap_file:
                    results_files.append(f"ASAP: {asap_file}")
            except Exception as e:
                logger.error(f"ASAP evaluation failed: {e}")
                print(f"ASAP evaluation failed: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"UNIFIED EVALUATION COMPLETED - Model: {model_info}")
    print(f"{'='*80}")
    print("Results files generated:")
    for result_file in results_files:
        print(f"  • {result_file}")
    print(f"  • Log: {log_filename}")

    logger.info("Unified evaluation completed successfully")


if __name__ == "__main__":
    main()
