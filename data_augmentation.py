#!/usr/bin/env python3
"""
Data Augmentation System for PerformanceMOS Model Training

This system creates corrupted MIDI performances with realistic mistakes
to provide a full range of scores for training the PerformanceMOS model.

Based on ASAP (max score 80) and ATEPP (max score 99) datasets.
"""

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import partitura as pt
from tqdm import tqdm
import warnings
from pathlib import Path
import hook

warnings.filterwarnings('ignore')

@dataclass
class CorruptionConfig:
    """Configuration for different types of performance corruptions.

    Matches Algorithm 1 in the paper: per performance, each corruption type
    is drawn independently as a Bernoulli with the probabilities below; if
    drawn, a penalty sampled from its range is applied. All draws compose
    additively on a single corrupted sample.
    """
    # Small mistake penalties (per error, with variability)
    wrong_notes_penalty_range: tuple = (-3, -1)
    timing_errors_penalty_range: tuple = (-2, -0.5)
    wrong_dynamics_penalty_range: tuple = (-1.5, -0.3)

    # Severe corruption penalties
    missing_voice_penalty_range: tuple = (-60, -30)
    missing_passage_penalty_range: tuple = (-50, -20)

    # Bernoulli probabilities per Algorithm 1
    p_wrong_notes: float = 0.5
    p_timing_errors: float = 0.5
    p_wrong_dynamics: float = 0.5
    p_missing_voice: float = 0.2
    p_missing_passage: float = 0.2

    # Intensity for the small-mistake corruptions
    corruption_intensity_range: tuple = (0.1, 1.0)

    # Base scores for datasets
    asap_max_score: int = 80
    atepp_max_score: int = 99


class DataAugmentationSystem:
    """System for generating corrupted performance data for MOS training"""
    
    def __init__(self, 
                 asap_dir: str = None,
                 atepp_dir: str = None,
                 output_dir: str = "augmented_data",
                 corruption_config: CorruptionConfig = None):
        self.asap_dir = asap_dir
        self.atepp_dir = atepp_dir
        self.output_dir = output_dir
        self.config = corruption_config or CorruptionConfig()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "midi"), exist_ok=True)
        
        # Initialize storage for corrupted performances
        self.corrupted_performances = []
        
    def load_asap_data(self) -> List[Tuple]:
        """Load ASAP dataset score-performance pairs"""
        if not self.asap_dir or not os.path.exists(self.asap_dir):
            print("ASAP directory not found, skipping...")
            return []
            
        print("Loading ASAP dataset...")
        note_pairs = []
        
        # Find all MIDI performance files
        midi_files = glob.glob(os.path.join(self.asap_dir, "**/*.mid"), recursive=True)
        
        for midi_path in tqdm(midi_files, desc="Loading ASAP files"):  # Limit for demonstration
            try:
                # Construct paths
                alignment_path = midi_path[:-4] + "_note_alignments/note_alignment.tsv"
                score_dir = os.path.dirname(midi_path)
                score_path = os.path.join(score_dir, "xml_score.musicxml")
                
                if not (os.path.exists(score_path) and os.path.exists(alignment_path)):
                    continue
                    
                # Load score
                score = pt.load_musicxml(score_path)
                score_part = score[0] if isinstance(score, list) else score.parts[0]
                
                # Load performance and alignment
                performance = pt.load_performance(midi_path)
                alignment = pt.io.importparangonada.load_alignment_from_ASAP(alignment_path)
                
                # Basic quality filter
                match_count = len([a for a in alignment if a['label'] == 'match'])
                total_count = len(alignment) 
                if total_count == 0 or (match_count / total_count) < 0.5:
                    continue
                    
                note_pairs.append({
                    'score_part': score_part,
                    'performance': performance,
                    'alignment': alignment,
                    'original_path': midi_path,
                    'dataset': 'asap',
                    'max_score': self.config.asap_max_score
                })
                
            except Exception as e:
                print(f"Error loading {midi_path}: {e}")
                continue
                
        print(f"Loaded {len(note_pairs)} ASAP performances")
        return note_pairs
    
    def load_atepp_data(self) -> List[Tuple]:
        """Load ATEPP dataset score-performance pairs"""
        if not self.atepp_dir or not os.path.exists(self.atepp_dir):
            print("ATEPP directory not found, skipping...")
            return []
            
        print("Loading ATEPP dataset...")
        note_pairs = []
        
        # Find all alignment files (ending with 'n.csv')
        alignment_files = glob.glob(os.path.join(self.atepp_dir, "**/[!z]*n.csv"), recursive=True)
        alignment_files = sorted(alignment_files)

        for align_path in tqdm(alignment_files[:800], desc="Loading ATEPP files"):  # Limit for demonstration
            try:
                # Construct paths
                midi_path = align_path[:-10] + ".mid"  # Remove "_align_n.csv"
                score_dir = os.path.dirname(os.path.dirname(align_path))  # Go up one directory
                score_files = glob.glob(os.path.join(score_dir, "*.*ml"))  # .xml, .musicxml
                
                if not score_files or not os.path.exists(midi_path):
                    continue
                    
                score_path = score_files[0]
                
                # Load components
                alignment = pt.io.importparangonada.load_parangonada_alignment(align_path)
                score = pt.load_musicxml(score_path, force_note_ids='keep')
                performance = pt.load_performance(midi_path)
                
                # Quality filter
                match_count = len([a for a in alignment if a['label'] == 'match'])
                total_count = len(alignment)
                if total_count == 0 or (match_count / total_count) < 0.5:
                    continue
                    
                note_pairs.append({
                    'score_part': score,
                    'performance': performance,  
                    'alignment': alignment,
                    'original_path': midi_path,
                    'dataset': 'atepp',
                    'max_score': self.config.atepp_max_score
                })
            except Exception as e:
                print(f"Error loading {align_path}: {e}")
                continue
                
        print(f"Loaded {len(note_pairs)} ATEPP performances")
        return note_pairs
    
    def introduce_wrong_notes_to_performance(self, performance, corruption_level: float = 0.1):
        """Introduce wrong notes directly into the performance"""
        try:
            # Get the performed part (assuming single part)
            if hasattr(performance, 'performedparts'):
                performed_part = performance.performedparts[0]
            else:
                performed_part = performance
            
            # Get note array from performed part
            note_array = performed_part.note_array()
            
            # Select random notes to corrupt with variable intensity
            n_corruptions = int(len(note_array) * corruption_level)
            if n_corruptions == 0:
                return performance, 0
                
            corruption_indices = np.random.choice(len(note_array), n_corruptions, replace=False)
            
            # Modify pitches directly in the note array with variable severity
            for idx in corruption_indices:
                # Variable shift based on corruption level (more corruption = bigger shifts)
                max_shift = int(3 * corruption_level) + 1
                shift = np.random.choice([-max_shift, -max_shift+1, -1, 1, max_shift-1, max_shift])
                note_array[idx]['pitch'] = np.clip(
                    note_array[idx]['pitch'] + shift, 21, 108  # Piano range
                )
            
            return performance, n_corruptions
            
        except Exception as e:
            print(f"Error introducing wrong notes: {e}")
            return performance, 0
    
    def introduce_timing_errors_to_performance(self, performance, corruption_level: float = 0.1):
        """Introduce timing irregularities and hesitations directly to performance"""
        try:
            # Get the performed part
            if hasattr(performance, 'performedparts'):
                performed_part = performance.performedparts[0]
            else:
                performed_part = performance
            
            # Get note array from performed part
            note_array = performed_part.note_array()
            
            n_corruptions = int(len(note_array) * corruption_level)
            if n_corruptions == 0:
                return performance, 0
                
            corruption_indices = np.random.choice(len(note_array), n_corruptions, replace=False)
            penalty_count = 0
            
            # Add timing jitter to note onsets with variable intensity
            for idx in corruption_indices:
                # Variable timing error based on corruption level
                max_error = 0.1 * corruption_level  # ±(0.01 to 0.1) seconds
                timing_error = np.random.uniform(-max_error, max_error)
                note_array[idx]['onset_sec'] = max(0, note_array[idx]['onset_sec'] + timing_error)
                penalty_count += 1
            
            # Add hesitations with variable frequency and duration
            hesitation_freq = 0.02 * corruption_level  # 0.2% to 2% hesitations
            hesitation_count = int(len(note_array) * hesitation_freq)
            if hesitation_count > 0:
                hesitation_indices = np.random.choice(len(note_array), hesitation_count, replace=False)
                
                for idx in hesitation_indices:
                    # Variable pause duration based on corruption level
                    pause_duration = np.random.uniform(0.2, 2.0) * corruption_level
                    note_array[idx]['onset_sec'] += pause_duration
                    penalty_count += 2  # Hesitations are worse
            
            return performance, penalty_count
            
        except Exception as e:
            print(f"Error introducing timing errors: {e}")
            return performance, 0
    
    def introduce_dynamic_errors_to_performance(self, performance, corruption_level: float = 0.15):
        """Introduce wrong dynamics and velocity errors directly to performance"""
        try:
            # Get the performed part
            if hasattr(performance, 'performedparts'):
                performed_part = performance.performedparts[0]
            else:
                performed_part = performance
            
            # Get note array from performed part
            note_array = performed_part.note_array()
            
            n_corruptions = int(len(note_array) * corruption_level)
            if n_corruptions == 0:
                return performance, 0
                
            corruption_indices = np.random.choice(len(note_array), n_corruptions, replace=False)
            
            # Modify velocities with variable intensity
            for idx in corruption_indices:
                # Variable velocity corruption based on corruption level
                original_vel = note_array[idx]['velocity']
                # More corruption = more extreme velocity changes
                vel_change = np.random.uniform(-50, 50) * corruption_level
                new_vel = int(np.clip(original_vel + vel_change, 20, 127))
                note_array[idx]['velocity'] = new_vel
            
            return performance, n_corruptions
            
        except Exception as e:
            print(f"Error introducing dynamic errors: {e}")
            return performance, 0
    
    def remove_voice_from_performance(self, performance, voice_to_remove: int = None):
        """Remove an entire voice from the performance (e.g., only right hand)"""
        try:
            # Get the performed part
            if hasattr(performance, 'performedparts'):
                performed_part = performance.performedparts[0]
            else:
                performed_part = performance
            
            # Get note array from performed part
            note_array = performed_part.note_array()
            
            # Find available voices
            if 'voice' not in note_array.dtype.names:
                # If no voice info, remove notes randomly (simulate missing hand)
                # Variable removal amount (30-70% of notes)
                removal_ratio = np.random.uniform(0.3, 0.7)
                n_to_remove = int(len(note_array) * removal_ratio)
                if n_to_remove > 0:
                    remove_indices = np.random.choice(len(note_array), n_to_remove, replace=False)
                    # Set velocity to 0 to effectively "remove" notes
                    note_array[remove_indices]['velocity'] = 0
                    return performance, 1
                return performance, 0
            
            unique_voices = np.unique(note_array['voice'])
            if len(unique_voices) <= 1:
                return performance, 0  # Can't remove if only one voice
                
            if voice_to_remove is None:
                voice_to_remove = np.random.choice(unique_voices)
            
            # Remove notes from the specified voice by setting velocity to 0
            voice_mask = note_array['voice'] == voice_to_remove
            note_array[voice_mask]['velocity'] = 0
            
            return performance, 1
            
        except Exception as e:
            print(f"Error removing voice: {e}")
            return performance, 0
    
    def remove_passage_from_performance(self, performance, passage_ratio: float = 0.3):
        """Remove a significant passage from the performance"""
        try:
            # Get the performed part
            if hasattr(performance, 'performedparts'):
                performed_part = performance.performedparts[0]
            else:
                performed_part = performance
            
            # Get note array from performed part
            note_array = performed_part.note_array()
            total_notes = len(note_array)
            
            # Variable passage removal (20-50% of notes)
            actual_passage_ratio = np.random.uniform(0.2, 0.5)
            passage_length = int(total_notes * actual_passage_ratio)
            if passage_length == 0:
                return performance, 0
                
            start_idx = np.random.randint(0, max(1, total_notes - passage_length))
            end_idx = start_idx + passage_length
            
            # Remove the passage by setting velocities to 0
            note_array[start_idx:end_idx]['velocity'] = 0
            
            return performance, 1
            
        except Exception as e:
            print(f"Error removing passage: {e}")
            return performance, 0
    
    def apply_corruptions(self, data_item: dict, corruptions: dict = None) -> dict:
        """Apply the corruption types flagged True in ``corruptions``.

        Implements one iteration of Algorithm 1: caller decides (via Bernoulli
        draws) which corruption types fire, this method composes them on a
        single sample and accumulates the penalty.

        Parameters
        ----------
        corruptions : dict[str, bool] | None
            Keys (any subset of) ``{wrong_notes, timing_errors,
            wrong_dynamics, missing_voice, missing_passage}``. Missing keys are
            treated as False. If None, all keys default to True (a "kitchen-
            sink" sample, useful for debugging).
        """
        corruptions = corruptions or {}
        corrupted_item = {
            'score_part': data_item['score_part'],
            'performance': data_item['performance'],
            'alignment': data_item['alignment'],
            'original_path': data_item['original_path'],
            'dataset': data_item['dataset'],
            'max_score': data_item['max_score']
        }
        total_penalty = 0
        corruption_description = []
        corrupted_performance = data_item['performance']

        if corruptions.get('wrong_notes'):
            intensity = np.random.uniform(*self.config.corruption_intensity_range)
            corrupted_performance, wrong_note_count = self.introduce_wrong_notes_to_performance(corrupted_performance, intensity)
            if wrong_note_count > 0:
                penalty_per_error = np.random.uniform(*self.config.wrong_notes_penalty_range)
                total_penalty += wrong_note_count * penalty_per_error
                corruption_description.append(f"wrong_notes({wrong_note_count})")

        if corruptions.get('timing_errors'):
            intensity = np.random.uniform(*self.config.corruption_intensity_range)
            corrupted_performance, timing_error_count = self.introduce_timing_errors_to_performance(corrupted_performance, intensity)
            if timing_error_count > 0:
                penalty_per_error = np.random.uniform(*self.config.timing_errors_penalty_range)
                total_penalty += timing_error_count * penalty_per_error
                corruption_description.append(f"timing_errors({timing_error_count})")

        if corruptions.get('wrong_dynamics'):
            intensity = np.random.uniform(*self.config.corruption_intensity_range)
            corrupted_performance, dynamic_error_count = self.introduce_dynamic_errors_to_performance(corrupted_performance, intensity)
            if dynamic_error_count > 0:
                penalty_per_error = np.random.uniform(*self.config.wrong_dynamics_penalty_range)
                total_penalty += dynamic_error_count * penalty_per_error
                corruption_description.append(f"dynamic_errors({dynamic_error_count})")

        if corruptions.get('missing_voice'):
            corrupted_performance, voice_removed = self.remove_voice_from_performance(corrupted_performance)
            if voice_removed:
                total_penalty += np.random.uniform(*self.config.missing_voice_penalty_range)
                corruption_description.append("missing_voice")

        if corruptions.get('missing_passage'):
            corrupted_performance, passage_removed = self.remove_passage_from_performance(corrupted_performance)
            if passage_removed:
                total_penalty += np.random.uniform(*self.config.missing_passage_penalty_range)
                corruption_description.append("missing_passage")

        original_score = data_item['max_score']
        final_score = max(1, original_score + total_penalty)

        active = [k for k, v in corruptions.items() if v]
        corrupted_item.update({
            'corrupted_performance': corrupted_performance,
            'original_score': original_score,
            'corrupted_score': final_score,
            'corruption_type': "+".join(active) if active else "clean",
            'corruption_description': ",".join(corruption_description),
            'penalty_applied': total_penalty
        })
        return corrupted_item
    
    def generate_augmented_data(self, n_augmentations_per_performance: int = 5):
        """Generate augmented dataset following Algorithm 1.

        For each performance, the clean version is kept as-is and
        ``n_augmentations_per_performance`` corrupted versions are produced.
        Each corrupted version draws every corruption type independently as a
        Bernoulli with the probabilities in ``CorruptionConfig`` (matching
        Algorithm 1); fired corruptions compose additively on the same sample.
        """
        print("Starting data augmentation process...")

        asap_data = self.load_asap_data() if self.asap_dir else []
        atepp_data = self.load_atepp_data() if self.atepp_dir else []

        all_data = asap_data + atepp_data
        print(f"Total performances to augment: {len(all_data)}")

        if not all_data:
            print("No data loaded! Please check your dataset paths.")
            return

        cfg = self.config
        augmented_data = []

        for data_item in tqdm(all_data, desc="Generating augmented performances"):
            clean_item = {
                'score_part': data_item['score_part'],
                'performance': data_item['performance'],
                'alignment': data_item['alignment'],
                'original_path': data_item['original_path'],
                'dataset': data_item['dataset'],
                'max_score': data_item['max_score'],
                'corrupted_performance': data_item['performance'],
                'original_score': data_item['max_score'],
                'corrupted_score': data_item['max_score'],
                'corruption_type': 'clean',
                'corruption_description': 'original',
                'penalty_applied': 0
            }
            augmented_data.append(clean_item)

            for _ in range(n_augmentations_per_performance):
                corruptions = {
                    'wrong_notes':     np.random.random() < cfg.p_wrong_notes,
                    'timing_errors':   np.random.random() < cfg.p_timing_errors,
                    'wrong_dynamics':  np.random.random() < cfg.p_wrong_dynamics,
                    'missing_voice':   np.random.random() < cfg.p_missing_voice,
                    'missing_passage': np.random.random() < cfg.p_missing_passage,
                }
                if not any(corruptions.values()):
                    # Algorithm 1 can produce an all-False draw (a duplicate of
                    # the clean sample); skip it to avoid biasing the score
                    # distribution toward R_max.
                    continue
                augmented_data.append(self.apply_corruptions(data_item, corruptions))
        
        print(f"Generated {len(augmented_data)} total performances (including originals)")
        self.corrupted_performances = augmented_data
        return augmented_data
    
    def save_augmented_dataset(self):
        """Save all augmented performances as MIDI files with CSV metadata"""
        if not self.corrupted_performances:
            print("No augmented data to save!")
            return
            
        print("Saving augmented dataset...")
        
        midi_dir = os.path.join(self.output_dir, "midi")
        csv_data = []
        
        for i, item in enumerate(tqdm(self.corrupted_performances, desc="Saving MIDI files")):
                # Generate filename
                original_name = os.path.basename(item['original_path']).replace('.mid', '')
                corruption_desc = item['corruption_description'].replace(',', '_')[:50]  # Limit length
                filename = f"{original_name}_{item['dataset']}_{corruption_desc}_{i:04d}.mid"
                midi_path = os.path.join(midi_dir, filename)
                
                # Save MIDI using partitura
                # performance_obj = pt.performance.Performance(
                #     id=f"augmented_{i}",
                #     performedparts=[item['corrupted_performance']] if hasattr(item['corrupted_performance'], 'note_array') else [item['corrupted_performance']]
                # )
                performance_obj = item['corrupted_performance'] 
                
                pt.save_performance_midi(performance_obj, midi_path)
                
                # Record metadata
                csv_data.append({
                    'filename': filename,
                    'original_path': item['original_path'],
                    'dataset': item['dataset'],
                    'original_score': item['original_score'],
                    'corrupted_score': item['corrupted_score'],
                    'corruption_type': item['corruption_type'],
                    'corruption_description': item['corruption_description'],
                    'penalty_applied': item['penalty_applied']
                })
                

        
        # Save CSV metadata
        csv_path = os.path.join(self.output_dir, "augmented_dataset_metadata.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Saved {len(csv_data)} MIDI files to {midi_dir}")
        print(f"Saved metadata to {csv_path}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total performances: {len(csv_data)}")
        print(f"Score distribution:")
        score_ranges = [
            (1, 20, "Very Poor"),
            (21, 40, "Poor"), 
            (41, 60, "Fair"),
            (61, 80, "Good"),
            (81, 99, "Excellent")
        ]
        
        for min_score, max_score, label in score_ranges:
            count = len(df[(df['corrupted_score'] >= min_score) & (df['corrupted_score'] <= max_score)])
            print(f"  {label} ({min_score}-{max_score}): {count} performances")
        
        corruption_stats = df['corruption_type'].value_counts()
        print(f"\nCorruption type distribution:")
        for corruption_type, count in corruption_stats.items():
            print(f"  {corruption_type}: {count}")


def main():
    """Main function to run data augmentation"""
    
    # Configuration - adjust paths as needed
    asap_dir = "/data/scratch/acw630/asap-dataset-alignment"
    atepp_dir = "/data/scratch/acw630/ATEPP-1.1" 
    output_dir = "augmented_performances"
    
    # Initialize system
    augmentation_system = DataAugmentationSystem(
        asap_dir=asap_dir if os.path.exists(asap_dir) else None,
        atepp_dir=atepp_dir if os.path.exists(atepp_dir) else None,
        output_dir=output_dir
    )
    
    # Generate augmented data
    augmented_data = augmentation_system.generate_augmented_data(
        n_augmentations_per_performance=8  # Generate 8 corrupted versions per original
    )
    
    # Save results
    if augmented_data:
        augmentation_system.save_augmented_dataset()
        print("Data augmentation completed successfully!")
    else:
        print("No augmented data generated. Please check your dataset paths.")


if __name__ == "__main__":
    main()
