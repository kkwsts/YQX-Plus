#!/usr/bin/env python3
"""
Metrics computation for YQX expressive performance evaluation
Computes objective metrics against Vienna4x22 dataset ground truth
"""

import os
import numpy as np
import pandas as pd
import partitura as pt
from typing import List, Tuple, Dict, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error
import glob
import json
import hook
from tqdm import tqdm
import parangonar
from parangonar.match import DualDTWNoteMatcher

import warnings
# warnings.filterwarnings('ignore')


from yqx import YQXSystem, get_matched_notes
from omegaconf import OmegaConf


def load_predicted_parameters(prediction_folder: str, vienna4x22_dir: str = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load predicted parameters from a test output folder
    
    Args:
        prediction_folder: Path to folder containing prediction results
        vienna4x22_dir: Path to Vienna4x22 dataset (needed for baseline models)
        
    Returns:
        Tuple of (predictions_dict, snote_id_dict) where snote_id_dict maps piece names to their note index arrays
    """
    predictions = {}
    snote_id = {}
    
    # Look for numpy prediction files first
    pred_files = glob.glob(os.path.join(prediction_folder, "*_predicted_parameters.npy"))
    
    for pred_file in pred_files:
        piece_name = os.path.basename(pred_file).replace('_predicted_parameters.npy', '')
        pred_params = np.load(pred_file)
        predictions[piece_name] = pred_params
        print(f"Loaded predictions for {piece_name}: {pred_params.shape}")
    
    # If no .npy files found, check for MIDI files (baseline models)
    if len(predictions) == 0:
        print(f"No .npy files found in {prediction_folder}, checking for MIDI files...")
        
        if vienna4x22_dir is None:
            print("Warning: vienna4x22_dir not provided, cannot process MIDI files")
            return predictions
        
        # Look for MIDI files
        midi_files = glob.glob(os.path.join(prediction_folder, "*.mid"))
        
        for midi_file in midi_files:
            # Extract piece name from MIDI filename
            filename = os.path.basename(midi_file)
            
            # Handle different naming conventions
            if filename.endswith('_expressive.mid'):
                piece_name = filename.replace('_expressive.mid', '')
            elif filename.endswith('_by_isgn_z0.mid'):
                piece_name = filename.replace('_by_isgn_z0.mid', '')
                # Handle case differences
                if piece_name.lower() == 'mozart6_k331_1st-mov':
                    piece_name = 'Mozart_K331_1st-mov'
                elif piece_name.lower() == 'chopin_op10_no3':
                    piece_name = 'Chopin_op10_no3'
                elif piece_name.lower() == 'chopin_op38':
                    piece_name = 'Chopin_op38'
                elif piece_name.lower() == 'schubert_d783_no15':
                    piece_name = 'Schubert_D783_no15'
            else:
                # Try to extract piece name by removing .mid extension
                piece_name = filename.replace('.mid', '')
            
            print(f"Processing MIDI file for {piece_name}: {midi_file}")
        
            # Load score
            score_path = os.path.join(vienna4x22_dir, "musicxml", f"{piece_name}.musicxml")
            if not os.path.exists(score_path):
                print(f"Warning: Score file not found: {score_path}")
                continue
            
            score_part = pt.load_musicxml(score_path)[0]
            score_note_array = score_part.note_array(include_grace_notes=True)
            
            # Load performance using pretty_midi and convert to partitura format
            import pretty_midi
            perf_midi = pretty_midi.PrettyMIDI(midi_file)
            pm_notes = perf_midi.instruments[0].notes
            
            # Convert pretty_midi notes to partitura note array format
            # partitura expects: (onset_sec, duration_sec, onset_tick, duration_tick, pitch, velocity, track, channel, id)
            perf_notes_list = []
            for i, note in enumerate(pm_notes):
                # Convert time to ticks (assuming 480 ticks per quarter note)
                ticks_per_quarter = 480
                # onset_tick = int(note.start * perf_midi.tempo_changes[0].tempo / 60 * ticks_per_quarter)
                # duration_tick = int((note.end - note.start) * perf_midi.tempo_changes[0].tempo / 60 * ticks_per_quarter)
                
                perf_notes_list.append((
                    note.start,           # onset_sec
                    note.end - note.start, # duration_sec
                    0,           # onset_tick
                    0,        # duration_tick
                    note.pitch,           # pitch
                    note.velocity,        # velocity
                    0,                    # track
                    0,                    # channel
                    f'n{i}'               # id
                ))
            
            # Create structured array with proper dtype
            dtype = [
                ('onset_sec', '<f4'), ('duration_sec', '<f4'), 
                ('onset_tick', '<i4'), ('duration_tick', '<i4'),
                ('pitch', '<i4'), ('velocity', '<i4'), 
                ('track', '<i4'), ('channel', '<i4'), ('id', '<U256')
            ]
            perf_note_array = np.array(perf_notes_list, dtype=dtype)
            
            # Perform alignment using parangonar DualDTWNoteMatcher
            print(f"Aligning {piece_name} with parangonar DualDTWNoteMatcher...")
            
            # Create note matcher
            note_matcher = DualDTWNoteMatcher()
            parangonar_matches = note_matcher(score_note_array, perf_note_array, process_ornaments=False, score_part=score_part)
            
            # Extract performance parameters
            parameters, piece_snote_id = pt.musicanalysis.encode_performance(score_note_array, perf_note_array, parangonar_matches)
            
            # Convert piece_snote_id to score note indices (same format as matched_note_idxs)
            # Find the actual indices in score_note_array for each note ID in piece_snote_id
            piece_snote_indices = np.array([np.where(score_note_array['id'] == note_id)[0][0] for note_id in piece_snote_id])
            
            
            # Create parameter array in same format as predictions
            # [beat_period, timing, velocity, articulation_log]
            pred_params = np.column_stack([
                parameters['beat_period'],
                parameters['timing'],
                parameters['velocity'],
                parameters['articulation_log']
            ])
            
            predictions[piece_name] = pred_params
            snote_id[piece_name] = piece_snote_indices
            print(f"Extracted parameters for {piece_name}: {pred_params.shape}")
            print(perf_note_array.shape, score_note_array.shape)
    
    return predictions, snote_id


def load_vienna4x22_ground_truth(vienna4x22_dir: str, piece_name: str) -> Dict[str, np.ndarray]:
    """
    Load ground truth parameters for a specific piece from Vienna4x22
    
    Args:
        vienna4x22_dir: Path to Vienna4x22 dataset
        piece_name: Name of the piece (e.g., 'Chopin_op10_no3')
        
    Returns:
        Dictionary mapping performance IDs to ground truth parameters
    """
    ground_truth = {}
    
    # Load score
    score_path = os.path.join(vienna4x22_dir, "musicxml", f"{piece_name}.musicxml")
    score_part = pt.load_musicxml(score_path)[0]
    score_notes = score_part.note_array()
    
    # Load all performances for this piece
    match_dir = os.path.join(vienna4x22_dir, "match")
    
    for i in range(1, 23):  # 22 performances
        match_file = os.path.join(match_dir, f"{piece_name}_p{i:02d}.match")
        
        if not os.path.exists(match_file):
            continue
            
        performed_part, alignment = pt.load_match(match_file)
        
        # Get matched notes
        pnote_array = performed_part.note_array()
        matched_note_idxs = get_matched_notes(score_notes, pnote_array, alignment)
        
        if len(matched_note_idxs) == 0:
            continue
            
        # Extract performance parameters
        parameters, _ = pt.musicanalysis.encode_performance(score_notes, pnote_array, alignment)
        
        # Get matched score notes and corresponding parameters
        matched_snote_array = score_notes[matched_note_idxs[:, 0]]
        
        # Create parameter array in same format as predictions
        # [beat_period, timing, velocity, articulation_log]
        gt_params = np.column_stack([
            parameters['beat_period'],
            parameters['timing'],
            parameters['velocity'],
            parameters['articulation_log']
        ])
        
        # Add note information and matched indices for filtering
        gt_data = {
            'parameters': gt_params,
            'pitch': matched_snote_array['pitch'],
            'onset_beat': matched_snote_array['onset_beat'],
            'duration_beat': matched_snote_array['duration_beat'],
            'voice': matched_snote_array['voice'],
            'matched_note_idxs': matched_note_idxs
        }
        
        ground_truth[f"p{i:02d}"] = gt_data
        

    
    print(f"Loaded {len(ground_truth)} ground truth performances for {piece_name}")
    return ground_truth


def filter_predictions_to_ground_truth(pred_params: np.ndarray, 
                                     gt_params: np.ndarray, gt_notes: Dict,
                                     matched_note_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter predicted parameters to match only the notes that exist in ground truth
    
    Args:
        pred_params: Predicted parameters array (full score)
        gt_params: Ground truth parameters array (matched notes only)
        gt_notes: Ground truth notes information (matched notes only)
        matched_note_idxs: Indices of score notes that were matched in performance
        
    Returns:
        Tuple of filtered predicted and ground truth parameters (should be same length)
    """
    # matched_note_idxs[:, 0] contains the score note indices that were matched
    score_note_indices = matched_note_idxs[:, 0]

    # print(pred_params.shape)
    # print(score_note_indices.max())
    # print(score_note_indices)
    
    # Filter predictions to only include the matched score notes
    filtered_pred_params = pred_params[score_note_indices]
    
    # Verify lengths match
    if len(filtered_pred_params) != len(gt_params):
        print(f"Warning: Length mismatch after filtering - filtered predictions: {len(filtered_pred_params)}, ground truth: {len(gt_params)}")
        return np.array([]), np.array([])
    
    return filtered_pred_params, gt_params


def filter_baselines_to_ground_truth(bl_params: np.ndarray, 
                                     gt_params: np.ndarray, gt_notes: Dict,
                                     matched_note_idxs: np.ndarray,
                                     snote_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter baseline parameters to match only the notes that exist in ground truth
    
    Args:
        bl_params: Baseline parameters array (already filtered from MIDI alignment)
        gt_params: Ground truth parameters array (matched notes only)
        gt_notes: Ground truth notes information (matched notes only)
        matched_note_idxs: Indices of score notes that were matched in performance
        snote_id: Score note indices that exist in baseline predictions (same format as matched_note_idxs)
    Returns:
        Tuple of filtered predicted and ground truth parameters (should be same length)
    """
    # matched_note_idxs[:, 0] contains the score note indices that were matched in ground truth
    gt_score_indices = matched_note_idxs[:, 0]
    
    # snote_id now contains the score note indices that exist in baseline predictions
    bl_score_indices = snote_id
    
    print(f"Baseline params shape: {bl_params.shape}")
    print(f"Ground truth params shape: {gt_params.shape}")
    print(f"Ground truth score indices: {len(gt_score_indices)} notes")
    print(f"Baseline score indices: {len(bl_score_indices)} notes")
    
    # Find intersection of score note indices that exist in both baseline and ground truth
    common_indices = np.intersect1d(gt_score_indices, bl_score_indices)

    
    # Get indices for baseline array
    bl_indices = np.where(np.isin(bl_score_indices, common_indices))[0]
    
    # Get indices for ground truth array  
    gt_indices = np.where(np.isin(gt_score_indices, common_indices))[0]
    
    # Filter both arrays to only include common notes
    filtered_bl_params = bl_params[bl_indices]
    filtered_gt_params = gt_params[gt_indices]
    
    print(f"Common notes: {len(common_indices)}")
    print(f"After filtering - baseline: {filtered_bl_params.shape}, ground truth: {filtered_gt_params.shape}")
    
    # Verify lengths match
    if len(filtered_bl_params) != len(filtered_gt_params):
        print(f"Warning: Length mismatch after filtering - baseline: {len(filtered_bl_params)}, ground truth: {len(filtered_gt_params)}")
        return np.array([]), np.array([])
    
    return filtered_bl_params, filtered_gt_params



def compute_rmse_metrics(pred_params: np.ndarray, gt_params: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE for each of the 4 target parameters
    
    Args:
        pred_params: Predicted parameters [N, 4]
        gt_params: Ground truth parameters [N, 4]
        
    Returns:
        Dictionary with RMSE for each parameter
    """
    param_names = ['beat_period', 'timing', 'velocity', 'articulation_log']
    metrics = {}
    
    for i, param_name in enumerate(param_names):
        rmse = np.sqrt(mean_squared_error(gt_params[:, i], pred_params[:, i]))
        metrics[f'rmse_{param_name}'] = rmse
    
    return metrics


def compute_curve_correlation(pred_params: np.ndarray, gt_params: np.ndarray,
                            pred_notes: Dict, gt_notes: Dict) -> Dict[str, float]:
    """
    Compute correlation for tempo and velocity curves at onset level
    
    Args:
        pred_params: Predicted parameters [N, 4]
        gt_params: Ground truth parameters [N, 4]
        pred_notes: Predicted notes information
        gt_notes: Ground truth notes information
        
    Returns:
        Dictionary with correlation coefficients
    """
    metrics = {}
    
    # Create onset-level curves
    def create_onset_curves(params, notes):
        onset_data = {}
        
        for i in range(len(notes['onset_beat'])):
            
            if i >= len(params):
                break
            
            onset = notes['onset_beat'][i]
            
            if onset not in onset_data:
                onset_data[onset] = {
                    'tempo': [],  # beat_period
                    'velocity': []
                }
            
            onset_data[onset]['tempo'].append(params[i, 0])  # beat_period
            onset_data[onset]['velocity'].append(params[i, 2])  # velocity
        
        # Average values at each onset
        onsets = sorted(onset_data.keys())
        tempo_curve = [np.mean(onset_data[onset]['tempo']) for onset in onsets]
        velocity_curve = [np.mean(onset_data[onset]['velocity']) for onset in onsets]
        
        return tempo_curve, velocity_curve, onsets
    
    # Create curves for predictions and ground truth
    pred_tempo, pred_velocity, pred_onsets = create_onset_curves(pred_params, pred_notes)
    gt_tempo, gt_velocity, gt_onsets = create_onset_curves(gt_params, gt_notes)
    
    # Find common onsets
    common_onsets = sorted(set(pred_onsets) & set(gt_onsets))
    
    if len(common_onsets) < 2:
        metrics['corr_tempo'] = np.nan
        metrics['corr_velocity'] = np.nan
        return metrics
    
    # Get values at common onsets
    pred_tempo_common = [pred_tempo[pred_onsets.index(onset)] for onset in common_onsets]
    pred_velocity_common = [pred_velocity[pred_onsets.index(onset)] for onset in common_onsets]
    gt_tempo_common = [gt_tempo[gt_onsets.index(onset)] for onset in common_onsets]
    gt_velocity_common = [gt_velocity[gt_onsets.index(onset)] for onset in common_onsets]
    
    # Compute correlations
    try:
        tempo_corr, _ = stats.pearsonr(pred_tempo_common, gt_tempo_common)
        velocity_corr, _ = stats.pearsonr(pred_velocity_common, gt_velocity_common)
    except:
        tempo_corr = np.nan
        velocity_corr = np.nan
    
    metrics['corr_tempo'] = tempo_corr
    metrics['corr_velocity'] = velocity_corr
    
    return metrics


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values
    
    Args:
        values: List of metric values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) == 0:
        return np.nan, np.nan
    
    # Remove NaN values
    values = [v for v in values if not np.isnan(v)]
    
    if len(values) == 0:
        return np.nan, np.nan
    
    mean_val = np.mean(values)
    std_err = stats.sem(values)
    
    # Compute confidence interval
    ci = stats.t.interval(confidence, len(values) - 1, loc=mean_val, scale=std_err)
    
    return ci[0], ci[1]


def evaluate_model_predictions(prediction_folder: str, vienna4x22_dir: str, 
                             config_path: str = 'config/default.yml') -> Dict:
    """
    Evaluate model predictions against Vienna4x22 ground truth
    
    Args:
        prediction_folder: Path to folder containing prediction results
        vienna4x22_dir: Path to Vienna4x22 dataset
        config_path: Path to configuration file
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating predictions from: {prediction_folder}")
    print(f"Using Vienna4x22 dataset: {vienna4x22_dir}")
    
    # Load configuration to get feature extractor
    config = OmegaConf.load(config_path)
    
    # Load predictions
    predictions, snote_id_dict = load_predicted_parameters(prediction_folder, vienna4x22_dir)
    
    # Define pieces
    pieces = ["Chopin_op10_no3", "Chopin_op38", "Mozart_K331_1st-mov", "Schubert_D783_no15"]
    
    # Initialize results storage
    all_metrics = {
        'rmse_beat_period': [],
        'rmse_timing': [],
        'rmse_velocity': [],
        'rmse_articulation_log': [],
        'corr_tempo': [],
        'corr_velocity': []
    }
    
    piece_results = {}
    
    for piece_name in pieces:
        print(f"\nProcessing {piece_name}...")
        
        # Load ground truth for this piece
        ground_truth = load_vienna4x22_ground_truth(vienna4x22_dir, piece_name)
        
        
        # Predictions are already loaded as numpy arrays - no need for note info
        pred_params = predictions[piece_name]
        piece_metrics = []
        
        # Compare against each ground truth performance
        for perf_id, gt_data in ground_truth.items():
            
            # Check if this is a baseline model (has snote_id for this piece)
            if piece_name in snote_id_dict:
                # For baseline models (MIDI-based), use baseline filtering
                filtered_pred, filtered_gt = filter_baselines_to_ground_truth(
                    pred_params, gt_data['parameters'], gt_data, gt_data['matched_note_idxs'], snote_id_dict[piece_name]
                )
            else:
                # For YQX models (full score predictions), use standard filtering
                filtered_pred, filtered_gt = filter_predictions_to_ground_truth(
                    pred_params, gt_data['parameters'], gt_data, gt_data['matched_note_idxs']
                )
                
            if len(filtered_pred) == 0:
                print(f"Warning: No filtered predictions found for {piece_name} {perf_id}")
                continue
                
            # Compute metrics
            rmse_metrics = compute_rmse_metrics(filtered_pred, filtered_gt)
            corr_metrics = compute_curve_correlation(filtered_pred, filtered_gt, gt_data, gt_data)
            
            # Combine metrics
            combined_metrics = {**rmse_metrics, **corr_metrics}
            combined_metrics['performance_id'] = perf_id
            combined_metrics['num_filtered_notes'] = len(filtered_pred)
                
            piece_metrics.append(combined_metrics)
            
            # Add to global metrics
            for metric_name in all_metrics.keys():
                if metric_name in combined_metrics:
                    all_metrics[metric_name].append(combined_metrics[metric_name])

        
        piece_results[piece_name] = piece_metrics
        print(f"Processed {len(piece_metrics)} performances for {piece_name}")
    
    # Compute overall statistics
    overall_results = {}
    
    for metric_name, values in all_metrics.items():
        if len(values) == 0:
            overall_results[metric_name] = {
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_samples': 0
            }
        else:
            # Remove NaN values
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) == 0:
                overall_results[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_samples': 0
                }
            else:
                mean_val = np.mean(clean_values)
                std_val = np.std(clean_values)
                ci_lower, ci_upper = compute_confidence_interval(clean_values)
                
                overall_results[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_samples': len(clean_values)
                }
    
    # Create final results dictionary
    results = {
        'overall_metrics': overall_results,
        'piece_results': piece_results,
        'prediction_folder': prediction_folder,
        'vienna4x22_dir': vienna4x22_dir,
        'total_performances_evaluated': sum(len(piece_metrics) for piece_metrics in piece_results.values())
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*80)
    print("YQX EXPRESSIVE PERFORMANCE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nPrediction folder: {results['prediction_folder']}")
    print(f"Total performances evaluated: {results['total_performances_evaluated']}")
    
    print("\nOVERALL METRICS (mean ± std, 95% CI):")
    print("-" * 60)
    
    metric_names = {
        'rmse_beat_period': 'RMSE Beat Period',
        'rmse_timing': 'RMSE Timing',
        'rmse_velocity': 'RMSE Velocity', 
        'rmse_articulation_log': 'RMSE Articulation',
        'corr_tempo': 'Tempo Correlation',
        'corr_velocity': 'Velocity Correlation'
    }
    
    for metric_key, metric_display in metric_names.items():
        metric_data = results['overall_metrics'][metric_key]
        
        if metric_data['n_samples'] == 0:
            print(f"{metric_display:20s}: No data")
        else:
            mean_val = metric_data['mean']
            std_val = metric_data['std']
            ci_lower = metric_data['ci_lower']
            ci_upper = metric_data['ci_upper']
            n_samples = metric_data['n_samples']
            
            if 'corr' in metric_key:
                # Correlation metrics
                print(f"{metric_display:20s}: {mean_val:.4f} ± {std_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (n={n_samples})")
            else:
                # RMSE metrics
                print(f"{metric_display:20s}: {mean_val:.4f} ± {std_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (n={n_samples})")
    
    print("\n" + "="*80)


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    """Main function for running evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YQX model predictions against Vienna4x22')
    parser.add_argument('--prediction_folder', type=str, required=True,
                      help='Path to folder containing prediction results')
    parser.add_argument('--vienna4x22_dir', type=str, 
                      default='/data/scratch/acw630/vienna4x22',
                      help='Path to Vienna4x22 dataset')
    parser.add_argument('--config', type=str, default='config/default.yml',
                      help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save results JSON (optional)')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model_predictions(
        prediction_folder=args.prediction_folder,
        vienna4x22_dir=args.vienna4x22_dir,
        config_path=args.config
    )
    
    # Print results
    print_results(results)
    
    # # Save results if requested
    # if args.output:
    #     save_results(results, args.output)
    # else:
    #     # Auto-save to prediction folder
    #     auto_output = os.path.join(args.prediction_folder, 'evaluation_results.json')
    #     save_results(results, auto_output)

    if args.output:
        prediction_folder_name = os.path.basename(os.path.normpath(args.prediction_folder))
        prediction_metrics_dir = os.path.join(args.output, prediction_folder_name)
        os.makedirs(prediction_metrics_dir, exist_ok=True)

        auto_output = os.path.join(prediction_metrics_dir, 'evaluation_results.json')
        save_results(results, auto_output)


if __name__ == "__main__":
    main()
