#!/usr/bin/env python3
"""
YQX: Expressive Music Performance System
Implementation based on the research paper by Widmer, Flossmann, and Grachten

This system learns to predict expressive performance parameters (timing, dynamics, articulation)
from musical score context using a Bayesian model trained on human performances.
"""

import os, time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import partitura as pt
from tqdm import tqdm
import pretty_midi
import hook
import warnings
warnings.filterwarnings('ignore')

from expressivenote import ExpressiveNote  
from gmm import BayesianExpressiveModel
from flow import FMExpressiveModel


class FeatureExtractor:
    """Extract musical context features from note arrays"""
    
    def __init__(self):
        self.ir_categories = [
            'Process', 'Reversal', 'Registral_Return', 'Intervallic_Duplication'
        ]
    
    def extract_melody(self, note_array: np.ndarray) -> np.ndarray:
        """Extract melody line (highest notes) from note array"""
        # Simple melody extraction: take highest note at each time point
        unique_onsets = np.unique(note_array['onset_beat'])
        melody_notes = []
        
        for onset in unique_onsets:
            notes_at_onset = note_array[note_array['onset_beat'] == onset]
            highest_note = notes_at_onset[np.argmax(notes_at_onset['pitch'])]
            melody_notes.append(highest_note)
        
        return np.array(melody_notes, dtype=note_array.dtype)
    
    def compute_rhythmic_context(self, durations: np.ndarray, idx: int) -> str:
        """Compute rhythmic context (e.g., 's-s-l' for short-short-long)"""
        if idx == 0 or idx >= len(durations) - 1:
            return "boundary"
        
        # Categorize durations relative to neighbors
        prev_dur = durations[idx-1]
        curr_dur = durations[idx]
        next_dur = durations[idx+1]
        
        def categorize_duration(dur, ref_dur):
            ratio = dur / ref_dur if ref_dur > 0 else 1.0
            if ratio < 0.75:
                return 's'  # short
            elif ratio > 1.33:
                return 'l'  # long
            else:
                return 'm'  # medium
        
        prev_cat = categorize_duration(prev_dur, curr_dur)
        next_cat = categorize_duration(next_dur, curr_dur)
        
        return f"{prev_cat}-{next_cat}"
    
    def compute_ir_analysis(self, pitches: np.ndarray, idx: int) -> Tuple[str, float]:
        """Simplified Implication-Realization analysis"""
        if idx == 0 or idx >= len(pitches) - 2:
            return "boundary", 0.0
        
        # Analyze melodic intervals
        int1 = pitches[idx] - pitches[idx-1]
        int2 = pitches[idx+1] - pitches[idx]
        
        # Simplified IR categorization
        if abs(int1) <= 2 and abs(int2) <= 2:
            ir_label = "Process"
            closure = -0.5  # Low closure (continuing)
        elif int1 * int2 < 0:  # Direction change
            ir_label = "Reversal"
            closure = 0.3  # Medium closure
        elif abs(int1) > 4 or abs(int2) > 4:  # Large intervals
            ir_label = "Registral_Return"
            closure = 0.7  # High closure
        else:
            ir_label = "Intervallic_Duplication"
            closure = 0.0
        
        return ir_label, closure
    
    def detect_phrase_boundaries(self, note_array: np.ndarray) -> List[int]:
        """Simple phrase boundary detection based on rests and large intervals"""
        boundaries = [0]
        
        for i in range(1, len(note_array)):
            # Check for rest (gap in onset times)
            prev_end = note_array[i-1]['onset_beat'] + note_array[i-1]['duration_beat']
            curr_start = note_array[i]['onset_beat']
            
            if curr_start - prev_end > 0.5:  # Rest longer than half beat
                boundaries.append(i)
            
            # Check for large pitch jump
            pitch_diff = abs(note_array[i]['pitch'] - note_array[i-1]['pitch'])
            if pitch_diff > 12:  # Octave or more
                boundaries.append(i)
        
        boundaries.append(len(note_array))
        return sorted(list(set(boundaries)))
    
    def extract_features(self, score_notes: np.ndarray, parameters: Optional[np.ndarray] = None) -> List[ExpressiveNote]:
        """Extract features from score and parameters (if available)"""
        melody = self.extract_melody(score_notes)
        phrase_boundaries = self.detect_phrase_boundaries(melody)

        expressive_notes = []
        
        for i, note in enumerate(melody):
            # Basic note information
            pitch = note['pitch']
            onset_beat = note['onset_beat']
            duration_beat = note['duration_beat']
            
            # Context features
            if i < len(melody) - 1:
                pitch_interval = melody[i+1]['pitch'] - pitch
                duration_ratio = melody[i+1]['duration_beat'] / duration_beat if duration_beat > 0 else 1.0
            else:
                pitch_interval = 0
                duration_ratio = 1.0
            
            rhythmic_context = self.compute_rhythmic_context(melody['duration_beat'], i)
            ir_label, ir_closure = self.compute_ir_analysis(melody['pitch'], i)
            
            # Position in phrase
            phrase_idx = 0
            for j, boundary in enumerate(phrase_boundaries[:-1]):
                if boundary <= i < phrase_boundaries[j+1]:
                    phrase_idx = j
                    break
            
            phrase_start = phrase_boundaries[phrase_idx]
            phrase_end = phrase_boundaries[phrase_idx + 1]
            position_in_phrase = (i - phrase_start) / max(1, phrase_end - phrase_start)
            
            # Expressive targets (if performance data available)
            beat_period = None
            timing = None
            velocity = None
            articulation_log = None
            
            if parameters is not None and i < len(parameters):
                perf_param = parameters[i]
                
                # Timing ratio (IOI ratio)
                beat_period = perf_param['beat_period'] 
                timing = perf_param['timing'] 
                
                # Loudness (velocity)
                velocity = perf_param['velocity']
                
                # Articulation ratio
                articulation_log = perf_param['articulation_log'] 
            
            expressive_note = ExpressiveNote(
                pitch=pitch,
                onset_beat=onset_beat,
                duration_beat=duration_beat,
                pitch_interval=pitch_interval,
                duration_ratio=duration_ratio,
                rhythmic_context=rhythmic_context,
                ir_label=ir_label,
                ir_closure=ir_closure,
                position_in_phrase=position_in_phrase,
                beat_period=beat_period,
                timing=timing,
                velocity=velocity,
                articulation_log=articulation_log
            )
            
            expressive_notes.append(expressive_note)
        
        return expressive_notes


class YQXSystem:
    """Complete YQX expressive performance system"""
    
    def __init__(self, data_dir: str, asap_dir: Optional[str] = None):
        self.data_dir = data_dir
        self.musicxml_dir = os.path.join(data_dir, "musicxml")
        self.match_dir = os.path.join(data_dir, "match")
        
        self.asap_dir = asap_dir
        if asap_dir is not None:
            self.asap_split_csv = os.path.join(asap_dir, "metadata-v1.3.csv")
            
        self.feature_extractor = FeatureExtractor()
        self.model = BayesianExpressiveModel()

    def get_asap_aligned_note_arrays(self, split='train', split_csv_path=None):
        """Load ASAP aligned note arrays with encoded performance parameters
        
        Args:
            split: 'train' or 'test' 
            split_csv_path: Path to CSV file containing difficulty_split_mid column for score-based splitting
        """
        if self.asap_dir is None:
            print("ASAP directory not provided")
            return []
        
        note_array_pairs = []
        
        # Load split information if provided
        split_performances = set()
        if split_csv_path and os.path.exists(split_csv_path):
            import pandas as pd
            split_df = pd.read_csv(split_csv_path)
            
            # Filter by the desired split (train/test) and extract MIDI performance paths
            split_rows = split_df[split_df['difficulty_split_mid'] == split]
            
            for _, row in split_rows.iterrows():
                if 'midi_performance' in row and pd.notna(row['midi_performance']):
                    # Add the relative path to our set
                    split_performances.add(row['midi_performance'])
            
            print(f"Found {len(split_performances)} performances in {split} split from CSV")
        else:
            print("Warning: No split CSV provided, this will cause data leakage!")
            return []

        
        # Process performances that are in the specified split
        for perf_path_relative in tqdm(split_performances):
            # Construct full path
            p_path = os.path.join(self.asap_dir, perf_path_relative)
            
            # Skip if file doesn't exist
            if not os.path.exists(p_path):
                continue
            
            # Construct paths based on ASAP structure
            alignment_path = p_path[:-4] + "_note_alignments/note_alignment.tsv"
            score_path = os.path.join("/".join(p_path.split("/")[:-1]), "xml_score.musicxml")
            
            # Check if required files exist
            if not (os.path.exists(score_path) and os.path.exists(alignment_path)):
                continue
            
            # Load score
            try:
                score = pt.load_musicxml(score_path)
                score_part = score[0] if isinstance(score, list) else score.parts[0]
                snote_array = score_part.note_array()
            except Exception as e:
                print(f"Error loading score {score_path}: {e}")
                continue
            
            # Load performance
            performance = pt.load_performance(p_path)
            alignment = pt.io.importparangonada.load_alignment_from_ASAP(alignment_path)

            
            # Filter out poor quality alignments
            match_aligns = [a for a in alignment if a['label'] == 'match']
            insertion_aligns = [a for a in alignment if a['label'] == 'insertion']
            deletion_aligns = [a for a in alignment if a['label'] == 'deletion']
            
            total_aligns = len(match_aligns) + len(insertion_aligns) + len(deletion_aligns)
            if total_aligns == 0 or (len(match_aligns) / total_aligns) < 0.5:
                print(f"Poor alignment quality for {p_path}, skipping...")
                continue
            
            # Check if score needs unfolding (based on alignment IDs)
            if (len(alignment) > 0 and 'score_id' in alignment[0] 
                and "-" in str(alignment[0]['score_id'])
                and "-" not in str(snote_array['id'][0])):
                # unfold the score if need 
                score_part = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts))
                snote_array = score_part.note_array()
            
            # Encode performance parameters using partitura
            parameters, snote_ids = pt.musicanalysis.encode_performance(
                score_part, performance, alignment
            )
            
            # Filter out invalid tempo (following original script's filter)
            avg_tempo = 60 / parameters['beat_period'].mean()
            if avg_tempo > 200:  # Skip if tempo is too fast
                continue
            
            # Get matched score notes
            matched_snote_array = snote_array[np.isin(snote_array['id'], snote_ids)]
            
            if len(matched_snote_array) > 0 and len(parameters) > 0:
                note_array_pairs.append((matched_snote_array, parameters))
            
        print(f"Loaded {len(note_array_pairs)} ASAP score-performance pairs for {split}")
        return note_array_pairs
    
    def get_v422_aligned_note_arrays(self, split='train'):
        """Load Vienna4x22 aligned note arrays"""
        note_array_pairs = []
        
        if split == 'train':
            pieces = ["Chopin_op10_no3", "Chopin_op38", "Mozart_K331_1st-mov", "Schubert_D783_no15"]
        else:
            pieces = []
        
        for piece_name in pieces:
            score_fn = os.path.join(self.musicxml_dir, f"{piece_name}.musicxml")
            if not os.path.exists(score_fn):
                print(f"Warning: Score file not found: {score_fn}")
                continue
                
            try:
                score_part = pt.load_musicxml(score_fn)[0]
            except Exception as e:
                print(f"Error loading score {score_fn}: {e}")
                continue
            
            for i in range(1, 23):  # 22 performances
                match_fn = os.path.join(self.match_dir, f"{piece_name}_p{i:02d}.match")
                if not os.path.exists(match_fn):
                    continue
                
                performed_part, alignment = pt.load_match(match_fn)
                
                snote_array = score_part.note_array()
                pnote_array = performed_part.note_array()
                matched_note_idxs = pt.musicanalysis.get_matched_notes(snote_array, pnote_array, alignment)
                parameters, _ = pt.musicanalysis.encode_performance(snote_array, pnote_array, alignment)
                
                matched_snote_array = snote_array[matched_note_idxs[:, 0]]
                matched_pnote_array = pnote_array[matched_note_idxs[:, 1]]
                
                note_array_pairs.append((matched_snote_array, parameters))
        
        return note_array_pairs


    
    def train(self):
        """Train the YQX system"""
        print("Loading training data...")
        note_pairs = self.get_v422_aligned_note_arrays('train')
        if self.asap_dir is not None:
            note_pairs.extend(self.get_asap_aligned_note_arrays('train', self.asap_split_csv))
        print(f"Loaded {len(note_pairs)} score-performance pairs")
        
        # Extract features for all performances
        training_notes = []
        for score_notes, perf_notes in note_pairs:
            expressive_notes = self.feature_extractor.extract_features(score_notes, perf_notes)
            training_notes.append(expressive_notes)
        
        # Train model
        t0 = time.time()
        self.model.train(training_notes)
        print(f"Training time: {time.time() - t0} seconds")
        
    
    def render_performance(self, musicxml_path: str, output_midi_path: str):
        """Render expressive performance of a MusicXML score"""
        print(f"Loading score: {musicxml_path}")
        
        # Load score
        try:
            score_part = pt.load_musicxml(musicxml_path)[0]
        except Exception as e:
            print(f"Error loading MusicXML: {e}")
            return
        
        score_notes = score_part.note_array()
        
        # Extract features
        print("Extracting features...")
        expressive_notes = self.feature_extractor.extract_features(score_notes)
        
        # Predict expressive parameters
        print("Predicting expressive parameters...")
        predicted_parameters = self.model.predict(expressive_notes)
        
        # Generate MIDI
        print("Generating MIDI...")
        self._generate_midi(predicted_parameters, score_part, output_midi_path)
        print(f"Expressive performance saved to: {output_midi_path}")
    
    def _generate_midi(self, predicted_parameters: List[ExpressiveNote], score_part: pt.score.Part, output_path: str):
        """Generate MIDI file using partitura's decode_performance"""
        
        # Get the score note array
        score_note_array = score_part.note_array()
        melody_indices = []
        
        # Find melody note indices in the full score
        melody = self.feature_extractor.extract_melody(score_note_array)
        for melody_note in melody:
            # Find corresponding index in full score
            matches = np.where((score_note_array['onset_beat'] == melody_note['onset_beat']) & 
                             (score_note_array['pitch'] == melody_note['pitch']))[0]
            if len(matches) > 0:
                melody_indices.append(matches[0])
        
        # Create performance parameter array for the full score
        # Initialize with default values
        performance_params = np.zeros(len(score_note_array), dtype=[
            ('beat_period', 'f4'),
            ('timing', 'f4'), 
            ('velocity', 'i4'),
            ('articulation_log', 'f4')
        ])
        
        # Set default values
        performance_params['beat_period'] = 0.5  # Default 120 BPM quarter note
        performance_params['timing'] = 0.0      # No timing deviation
        performance_params['velocity'] = 0.5     # Medium velocity
        performance_params['articulation_log'] = 0.0  # No articulation change
        
        # Apply predicted expressive parameters to melody notes
        for i, note in enumerate(predicted_parameters):
            if i < len(melody_indices):
                score_idx = melody_indices[i]
                
                if note.beat_period is not None:
                    performance_params[score_idx]['beat_period'] = note.beat_period
                if note.timing is not None:
                    performance_params[score_idx]['timing'] = note.timing
                if note.velocity is not None:
                    performance_params[score_idx]['velocity'] = note.velocity
                if note.articulation_log is not None:
                    performance_params[score_idx]['articulation_log'] = note.articulation_log
        
        # For non-melody notes, use interpolated or default values
        # This is a simple approach - could be improved with better accompaniment modeling
        for i in range(len(score_note_array)):
            if i not in melody_indices:
                # Use nearest melody note's beat_period and timing
                if len(melody_indices) > 0:
                    onset = score_note_array[i]['onset_beat']
                    melody_onsets = score_note_array[melody_indices]['onset_beat']
                    nearest_idx = np.argmin(np.abs(melody_onsets - onset))
                    nearest_melody_idx = melody_indices[nearest_idx]
                    
                    performance_params[i]['beat_period'] = performance_params[nearest_melody_idx]['beat_period']
                    performance_params[i]['timing'] = performance_params[nearest_melody_idx]['timing'] * 0.5  # Reduced timing for accompaniment
                    # Keep default velocity and articulation for accompaniment
        
        # try:
        # Use partitura's decode_performance to create performed part
        performed_part = pt.musicanalysis.decode_performance(
            score_part, 
            performance_params,
            return_performance_array=False
        )
        
        # Create performance object
        performance = pt.performance.Performance(
            id="yqx_performance",
            performedparts=[performed_part]
        )
        
        # Save as MIDI
        pt.save_performance_midi(performance, output_path)
        
        # except Exception as e:
        #     print(f"Error using decode_performance: {e}")
        #     print("Falling back to manual MIDI generation...")
            
        #     # Fallback: manual MIDI generation using pretty_midi
        #     midi = pretty_midi.PrettyMIDI()
        #     piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
            
        #     # Apply expressive timing manually
        #     current_time = 0.0
            
        #     for i, score_note in enumerate(score_note_array):
        #         # Get performance parameters for this note
        #         beat_period = performance_params[i]['beat_period']
        #         timing = performance_params[i]['timing']
        #         velocity = int(performance_params[i]['velocity'])
        #         articulation_log = performance_params[i]['articulation_log']
                
        #         # Calculate onset time with expressive timing
        #         if i > 0:
        #             # Inter-onset interval with timing deviation
        #             score_ioi = (score_note['onset_beat'] - score_note_array[i-1]['onset_beat']) * beat_period
        #             expressive_ioi = score_ioi * (1.0 + timing)
        #             current_time += expressive_ioi
        #         else:
        #             current_time = score_note['onset_beat'] * beat_period
                
        #         # Calculate duration with articulation
        #         score_duration = score_note['duration_beat'] * beat_period
        #         expressive_duration = score_duration * np.exp(articulation_log)
                
        #         # Create MIDI note
        #         midi_note = pretty_midi.Note(
        #             velocity=np.clip(velocity, 1, 127),
        #             pitch=score_note['pitch'],
        #             start=current_time,
        #             end=current_time + expressive_duration
        #         )
                
        #         piano.notes.append(midi_note)
            
        #     midi.instruments.append(piano)
        #     midi.write(output_path)
        
    def save_model(self, filepath: str):
        """Save trained model"""
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model.load(filepath)

# Example usage
def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YQX Expressive Music Performance System')
    parser.add_argument('--data_dir', type=str, 
                      default="/Users/huanzhang/01Acdemics/PhD/Research/Datasets/vienna4x22",
                      help='Path to Vienna4x22 dataset directory')
    parser.add_argument('--asap_dir', type=str, 
                      default="/Users/huanzhang/01Acdemics/PhD/Research/Datasets/asap-dataset-alignment",
                      help='Path to ASAP dataset directory (optional)')
    parser.add_argument('--input_score', type=str,
                      default="/Users/huanzhang/01Acdemics/PhD/Research/Datasets/vienna4x22/musicxml/Chopin_op10_no3.musicxml",
                      help='Path to input MusicXML score for rendering')
    parser.add_argument('--output_midi', type=str,
                      default="Chopin_op10_no3_yqx.mid",
                      help='Path for output MIDI performance')
    parser.add_argument('--model_path', type=str, default='yqx_model.pkl',
                      help='Path to save/load model (default: yqx_model.pkl)')
    parser.add_argument('--train', action='store_true',
                      help='Train the model')
    parser.add_argument('--render', action='store_true',
                      help='Render a performance from input score')
    parser.add_argument('--use_asap', action='store_true',
                      help='Use ASAP dataset during training')
    
    args = parser.parse_args()
    
    # Initialize system with ASAP only if use_asap is True
    asap_dir = args.asap_dir if args.use_asap else None
    yqx = YQXSystem(args.data_dir, asap_dir)
    
    # Train if requested
    if args.train:
        print("Training YQX system...")
        yqx.train()
        yqx.save_model(args.model_path)
        print(f"Model saved to {args.model_path}")
    
    # Render if requested
    if args.render:
        if not os.path.exists(args.input_score):
            print(f"Error: Input score {args.input_score} not found")
            return
        
        yqx.load_model(args.model_path)
            
        print(f"Rendering performance of {args.input_score}")
        yqx.render_performance(args.input_score, args.output_midi)
        print(f"Performance saved to {args.output_midi}")

if __name__ == "__main__":
    main()