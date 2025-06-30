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
import partitura as pt
from tqdm import tqdm
import matplotlib.pyplot as plt 
import hook
import warnings
from omegaconf import OmegaConf, DictConfig
warnings.filterwarnings('ignore')

from expressivenote import ExpressiveNote  
from gmm import BayesianExpressiveModel
from flow import FMExpressiveModel
from features import FeatureExtractor


def get_matched_notes(spart_note_array, ppart_note_array, alignment):
    """
    Get the indices of the matched notes in an alignment

    Parameters
    ----------
    spart_note_array : structured numpy array
        note_array of the score part
    ppart_note_array : structured numpy array
        note_array of the performed part
    alignment : list
        The score--performance alignment, a list of dictionaries.
        (see `partitura.io.importmatch.alignment_from_matchfile` for reference)

    Returns
    -------
    matched_idxs : np.ndarray
        A 2D array containing the indices of the matched score and
        performed notes, where the columns are
        (index_in_score_note_array, index_in_performance_notearray)
    """
    # Get matched notes
    matched_idxs = []
    for al in alignment:
        # Get only matched notes (i.e., ignore inserted or deleted notes)
        if al["label"] == "match":
            # if ppart_note_array['id'].dtype != type(al['performance_id']):
            if not isinstance(ppart_note_array["id"], type(al["performance_id"])):
                p_id = str(al["performance_id"])
            else:
                p_id = al["performance_id"]

            p_idx = np.where(ppart_note_array["id"] == p_id)[0]

            s_idx = np.where(spart_note_array["id"] == al["score_id"])[0]

            if len(s_idx) > 0 and len(p_idx) > 0:
                s_idx = int(s_idx)
                p_idx = int(p_idx)
                matched_idxs.append((s_idx, p_idx))

    if len(matched_idxs) == 0:
        warnings.warn(
            "No matched note IDs found. "
            "Either the alignment contains no matches "
            "or the IDs in score of performance do not correspond to the alignment "
            "(maybe due to repeat unfolding)."
        )

    return np.array(matched_idxs)



class YQXSystem:
    """Complete YQX expressive performance system"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Set up paths
        self.data_dir = config.data.vienna4x22_dir
        self.musicxml_dir = os.path.join(self.data_dir, "musicxml")
        self.match_dir = os.path.join(self.data_dir, "match")
        
        self.asap_dir = config.data.asap_dir if config.data.use_asap else None
        if self.asap_dir is not None:
            self.asap_split_csv = os.path.join(self.asap_dir, "metadata-v1.3.csv")
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        
        # Initialize model based on config
        if config.model.type == "gmm":
            self.model = BayesianExpressiveModel(n_components=config.model.n_components)
        elif config.model.type == "flow":
            self.model = FMExpressiveModel()
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        
        # Create artifacts directory
        os.makedirs(config.output.artifacts_dir, exist_ok=True)

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
                matched_note_idxs = get_matched_notes(snote_array, pnote_array, alignment)
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
        
        print("Extracting features for training data...")
        training_notes, avg_tempos = [], []
        for idx, (score_notes, perf_params) in tqdm(enumerate(note_pairs)):
            
            # filter out outliers and standardize the time parameters to 120 bpm
            score_notes, perf_params, avg_tempo = self.feature_extractor.standardize_targets(score_notes, perf_params)
            avg_tempos.append(avg_tempo)
            
            expressive_notes = self.feature_extractor.extract_features(
                score_notes, 
                perf_params, 
                use_midihum_features=self.config.model.use_midihum_features
            )
            
            if idx == 0 and self.config.output.plot_targets:
                plot_path = os.path.join(self.config.output.artifacts_dir, "train_params.png")
                self.feature_extractor.plot_targets(expressive_notes, plot_path)
            
            training_notes.append(expressive_notes)
        
        if self.config.output.save_distributions:
            # Save the distribution of the training notes targets and avg_tempo
            targets_distribution = np.array([note.get_targets() for note_list in training_notes for note in note_list])
            np.save(os.path.join(self.config.output.artifacts_dir, "targets_distribution.npy"), targets_distribution)
            np.save(os.path.join(self.config.output.artifacts_dir, "avg_tempos.npy"), np.array(avg_tempos))
        
        # Train model
        t0 = time.time()
        self.model.train(training_notes, self.feature_extractor)
        print(f"Training time: {time.time() - t0} seconds")
        
        # Save model and encoders
        self.save_model(self.config.model.model_path)
    
    def render_performance(self, musicxml_path: str = None, output_midi: str = None, initial_tempo: int = None):
        """Render expressive performance of a MusicXML score"""
        print(f"Loading score: {musicxml_path}")
        
        # Load score
        score_part = pt.load_musicxml(musicxml_path)[0]
        
        score_notes = score_part.note_array()
        
        # Extract features
        print("Extracting features...")
        expressive_notes = self.feature_extractor.extract_features(
            score_notes, 
            use_midihum_features=self.config.model.use_midihum_features
        )
        
        # Predict expressive parameters
        print("Predicting expressive parameters...")
        predicted_expressive_notes = self.model.predict(expressive_notes, self.feature_extractor)
        
        plot_path = os.path.join(self.config.output.artifacts_dir, "pred_params.png")
        self.feature_extractor.plot_targets(predicted_expressive_notes, plot_path)
        
        # Generate MIDI
        print("Generating MIDI...")
        self._generate_midi(predicted_expressive_notes, score_part, output_midi, initial_tempo)
        print(f"Expressive performance saved to: {output_midi}")
    
    def _generate_midi(self, predicted_expressive_notes: List[ExpressiveNote], 
                       score_part: pt.score.Part, output_path: str, initial_tempo: float = 120):
        """Generate MIDI file using partitura's decode_performance"""
        
        # Get the score note array
        score_note_array = score_part.note_array()
        
        # Create performance parameter array for the full score
        # Initialize with default values
        performance_params = np.zeros(len(score_note_array), dtype=[
            ('beat_period', 'f4'),
            ('timing', 'f4'), 
            ('velocity', 'f4'),
            ('articulation_log', 'f4')
        ])
        

        # Apply predicted expressive parameters by matching notes
        for pred_note in predicted_expressive_notes:
            # Find matching note in score array
            matches = np.where(
                (score_note_array['onset_beat'] == pred_note.onset_beat) & 
                (score_note_array['pitch'] == pred_note.pitch)
            )[0]
            
            if len(matches) > 0:
                score_idx = matches[0]  # Take first match if multiple exist
                
                if pred_note.beat_period is not None:
                    performance_params[score_idx]['beat_period'] = pred_note.beat_period
                if pred_note.timing is not None:
                    performance_params[score_idx]['timing'] = pred_note.timing
                if pred_note.velocity is not None:
                    performance_params[score_idx]['velocity'] = pred_note.velocity
                if pred_note.articulation_log is not None:
                    performance_params[score_idx]['articulation_log'] = pred_note.articulation_log
        
        # scale the time parameters with user provided tempo
        performance_params['beat_period'] = performance_params['beat_period'] / (initial_tempo / 120)
        performance_params['timing'] = performance_params['timing'] / initial_tempo / 120
        
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
        

        
    def save_model(self, filepath: str = None):
        """Save trained model and feature encoders"""
        filepath = filepath or self.config.model.model_path
        self.model.save(filepath)
        self.feature_extractor.save_encoders(filepath + ".encoders")
    
    def load_model(self, filepath: str = None):
        """Load trained model and feature encoders"""
        filepath = filepath or self.config.model.model_path
        self.model.load(filepath)
        self.feature_extractor.load_encoders(filepath + ".encoders")

# Example usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YQX Expressive Music Performance System')
    parser.add_argument('--config', type=str, default='config/default.yml',
                      help='Path to configuration file')
    parser.add_argument('--override', type=str,
                      help='Path to override configuration file')
    parser.add_argument('overrides', nargs='*', 
                      help='Any key=value overrides for config')
    
    args = parser.parse_args()
    
    # Load base config
    conf = OmegaConf.load(args.config)
    
    # Load and merge override config if provided
    if args.override:
        override_conf = OmegaConf.load(args.override)
        conf = OmegaConf.merge(conf, override_conf)
    
    # Apply command line overrides
    if args.overrides:
        cli_conf = OmegaConf.from_cli(args.overrides)
        conf = OmegaConf.merge(conf, cli_conf)
    
    # Initialize system
    yqx = YQXSystem(conf)
    
    if conf.train.enabled:
        print("Training YQX system...")
        yqx.train()
    
    if conf.render.enabled:
        if not os.path.exists(conf.render.input_score):
            print(f"Error: Input score {conf.render.input_score} not found")
            return
            
        if not conf.train.enabled:
            yqx.load_model()


        print(f"Rendering performance of {conf.render.input_score}")
        yqx.render_performance(
            musicxml_path=conf.render.input_score,
            output_midi=conf.render.output_midi,
            initial_tempo=conf.render.initial_tempo
        )


if __name__ == "__main__":
    main()