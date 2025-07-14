#!/usr/bin/env python3
"""
YQX: Expressive Music Performance System
Implementation based on the research paper by Widmer, Flossmann, and Grachten

This system learns to predict expressive performance parameters (timing, dynamics, articulation)
from musical score context using a Bayesian model trained on human performances.
"""

import os, time, glob, pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import partitura as pt
from tqdm import tqdm
import hook
import warnings
from omegaconf import OmegaConf, DictConfig
import torch
import wandb
from torchinfo import summary
warnings.filterwarnings('ignore')

from expressivenote import ExpressiveNote  
from gmm import BayesianExpressiveModel
from bvae import BVAEExpressiveModel
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
        
        self.use_vienna4x22 = config.data.use_vienna4x22
        self.asap_dir = config.data.asap_dir if config.data.use_asap else None
        if self.asap_dir is not None:
            self.asap_split_csv = os.path.join(self.asap_dir, "metadata-v1.3.csv")
        
        self.atepp_dir = config.data.atepp_dir if config.data.use_atepp else None
        if self.atepp_dir is not None:
            self.atepp_meta_csv = config.data.atepp_meta_csv
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        
        self.context_features, self.targets = self.load_data()

        device = config.model.get('device', 'cpu')
        if device.startswith('cuda:'):
            # Check if specified GPU is available
            gpu_id = int(device.split(':')[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_id} not available, using CPU instead")
                device = "cpu"

        # Initialize model based on config
        if config.model.type == "gmm":
            gmm_config = config.model.gmm
            self.model = BayesianExpressiveModel(
                n_components=gmm_config.n_components,
                random_state=gmm_config.get('random_state', 42)
            )
        elif config.model.type == "flow":
            from flow_JASCO import FMExpressiveModel
            flow_config = config.model.flow
            self.model = FMExpressiveModel(
                features_dim=self.context_features.shape[1],
                target_dim=self.targets.shape[1], 
                hidden_dim=flow_config.get('hidden_dim', 128),
                use_midihum=config.model.use_midihum_features,
                num_heads=flow_config.get('num_heads', 4),
                num_layers=flow_config.get('num_layers', 2),
                flow_matcher_type=flow_config.get('flow_matcher_type', 'standard'),
                sigma=flow_config.get('sigma', 0.01),
                device=device
            )
        elif config.model.type == "bvae":
            bvae_config = config.model.bvae
            self.model = BVAEExpressiveModel(
                context_dim= self.context_features.shape[1],
                target_dim= self.targets.shape[1],
                latent_dim=bvae_config.get('latent_dim', 64),
                hidden_dims=bvae_config.get('hidden_dims', [256, 128]),
                beta=bvae_config.get('beta', 4.0),
                gamma=bvae_config.get('gamma', 1000.0),
                use_midihum=config.model.use_midihum_features,
                learning_rate=bvae_config.get('learning_rate', 0.001),
                device= device
            )
            
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        
        
        # Generate model path for experiment tracking
        self.model_path = self._generate_model_path(config)
        
        # Create artifacts directory
        os.makedirs(config.output.artifacts_dir, exist_ok=True)
        os.makedirs(config.output.ckpt_dir, exist_ok=True)
        
    def _generate_model_identifier(self, config: DictConfig) -> str:
        """Generate a consistent model identifier for experiment tracking"""
        parts = []
        
        # Add experiment name if provided
        if config.output.get('experiment_name'):
            parts.append(config.output.experiment_name)
        
        # Add model type
        parts.append(config.model.type)
        
        # Add model-specific parameters if requested
        if config.output.get('include_model_params', True):
            if config.model.type == "gmm":
                parts.append(f"nc{config.model.gmm.n_components}")
            elif config.model.type == "flow":
                parts.append(f"hd{config.model.flow.get('hidden_dim', 128)}")
                parts.append(f"nh{config.model.flow.get('num_heads', 4)}")
                parts.append(f"nl{config.model.flow.get('num_layers', 2)}")
                if config.model.flow.get('flow_matcher_type', 'standard') != 'standard':
                    parts.append(f"fmt{config.model.flow.flow_matcher_type}")
            elif config.model.type == "bvae":
                parts.append(f"ld{config.model.bvae.get('latent_dim', 64)}")
                parts.append(f"hd{config.model.bvae.get('hidden_dims', [256, 128])}")
                parts.append(f"b{config.model.bvae.get('beta', 4.0)}")
                parts.append(f"g{config.model.bvae.get('gamma', 10.0)}")
        
        # Add midihum features flag
        if config.model.use_midihum_features:
            parts.append("midihum")
        
        return "_".join(parts)
    
    def _generate_model_path(self, config: DictConfig) -> str:
        """Generate model file path"""
        identifier = self._generate_model_identifier(config)
        # Use configurable extension, default to .pkl for backward compatibility
        extension = config.output.get('model_extension', '.pkl')
        if not extension.startswith('.'):
            extension = '.' + extension
        filename = identifier + extension
        return os.path.join(config.output.ckpt_dir, filename)



    def get_asap_aligned_note_arrays(self, split='train', split_csv_path=None):
        """Load ASAP aligned note arrays with encoded performance parameters
        
        Args:
            split: 'train' or 'test' 
            split_csv_path: Path to CSV file containing difficulty_split_mid column for score-based splitting
        """
        if self.asap_dir is None:
            print("ASAP directory not provided")
            return []
        
        cache_file = os.path.join(self.config.output.artifacts_dir, f"asap_{split}_cache.pkl")
        if os.path.exists(cache_file):
            print(f"Loading ASAP {split} data from cache...")
            with open(cache_file, 'rb') as f:
                note_array_pairs = pickle.load(f)
            print(f"Loaded {len(note_array_pairs)} ASAP score-performance pairs from cache")
            return note_array_pairs
        
        print(f"Computing ASAP {split} data (will be cached for future use)...")
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
            
        with open(cache_file, 'wb') as f:
            pickle.dump(note_array_pairs, f)
        print(f"Saved {len(note_array_pairs)} ASAP score-performance pairs to cache")
        
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

    def get_atepp_aligned_note_arrays(self, split='train', split_csv_path=None):
        """Load ATEPP aligned note arrays with encoded performance parameters
        
        Args:
            split: 'train' or 'test'
            split_csv_path: Path to CSV file containing split information (optional)
        """
        if self.atepp_dir is None:
            print("ATEPP directory not provided")
            return []
        
        cache_file = os.path.join(self.config.output.artifacts_dir, f"atepp_{split}_cache.pkl")
        if os.path.exists(cache_file):
            print(f"Loading ATEPP {split} data from cache...")
            with open(cache_file, 'rb') as f:
                note_array_pairs = pickle.load(f)
            print(f"Loaded {len(note_array_pairs)} ATEPP score-performance pairs from cache")
            return note_array_pairs
        
        print(f"Computing ATEPP {split} data (will be cached for future use)...")
        note_array_pairs = []
        
        # Find all alignment files (ending with 'n.csv')
        alignment_paths = glob.glob(os.path.join(self.atepp_dir, "**/[!z]*n.csv"), recursive=True)
        alignment_paths = sorted(alignment_paths)
        
        # For each alignment file, find corresponding performance and score files
        for a_path in tqdm(alignment_paths, desc=f"Loading ATEPP {split} data"):
            try:
                # Construct performance path by changing extension
                p_path = a_path[:-10] + ".mid"  # Remove "_align_n.csv" and add ".mid"
                
                # Find score file (with .*l extension - .xml, .musicxml, .krn, etc.)
                score_dir = "/".join(p_path.split("/")[:-1])
                score_files = glob.glob(os.path.join(score_dir, "*.*l"))
                
                if not score_files:
                    continue
                s_path = score_files[0]  # Take the first match
                
                # Check if all required files exist
                if not (os.path.exists(s_path) and os.path.exists(a_path) and os.path.exists(p_path)):
                    continue
                
                # Load alignment and check quality
                alignment = pt.io.importparangonada.load_parangonada_alignment(a_path)
                
                # Filter out poor quality alignments
                match_aligns = [a for a in alignment if a['label'] == 'match']
                insertion_aligns = [a for a in alignment if a['label'] == 'insertion']
                deletion_aligns = [a for a in alignment if a['label'] == 'deletion']
                
                total_aligns = len(match_aligns) + len(insertion_aligns) + len(deletion_aligns)
                if total_aligns == 0 or (len(match_aligns) / total_aligns) < 0.5:
                    continue
                
                # Load score with force_note_ids='keep' for ATEPP
                score = pt.load_musicxml(s_path, force_note_ids='keep')  
                # if doesn't match the note id in alignment, unfold the score.
                if (('score_id' in alignment[0]) 
                    and ("-" in alignment[0]['score_id'])
                    and ("-" not in score.note_array(include_divs_per_quarter=False)['id'][0])): 
                    score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
                
                for a in alignment:
                    if 'score_id' in a:
                        a['score_id'] = a['score_id'].replace("P00_", "")

                # Load performance
                performance = pt.load_performance(p_path)
                
                # Encode performance parameters
                parameters, snote_ids = pt.musicanalysis.encode_performance(
                    score, performance, alignment
                )
                
                # Filter out invalid tempo (following original script's filter)
                avg_tempo = 60 / parameters['beat_period'].mean()
                if avg_tempo > 200:  # Skip if tempo is too fast
                    continue
                
                # Get score note array and filter to matched notes
                snote_array = score.note_array()
                matched_snote_array = snote_array[np.isin(snote_array['id'], snote_ids)]
                
                # sometimes the matched_snotes_array are shorter after the filtering (some id misfound?) so
                # we need to remove the corresponding parameters in the parameters array
                if len(matched_snote_array) != len(parameters):
                    # Get the IDs that were actually found in the score
                    matched_ids = matched_snote_array['id']
                    
                    # Check for duplicates in snote_ids
                    unique_snote_ids, unique_indices = np.unique(snote_ids, return_index=True)
                    if len(unique_snote_ids) != len(snote_ids):
                        print(f"Warning: Found {len(snote_ids) - len(unique_snote_ids)} duplicate IDs in snote_ids")
                        # Use only the first occurrence of each ID
                        snote_ids = np.array(snote_ids)[unique_indices]
                        parameters = parameters[unique_indices]
                    
                    # Now filter to only include parameters for IDs that exist in the score
                    present_mask = np.isin(snote_ids, matched_ids)
                    parameters = parameters[present_mask]
                
                    assert len(matched_snote_array) == len(parameters), "Length mismatch after filtering"
                
                if len(matched_snote_array) > 0 and len(parameters) > 0:
                    note_array_pairs.append((matched_snote_array, parameters))
            except Exception as e:
                print(f"Error processing ATEPP file {a_path}: {e}")
                continue
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(note_array_pairs, f)
        print(f"Saved {len(note_array_pairs)} ATEPP score-performance pairs to cache")
        
        print(f"Loaded {len(note_array_pairs)} ATEPP score-performance pairs for {split}")
        return note_array_pairs


    def get_dataset_features(self, note_pairs, dataset_name: str, 
                                       use_midihum: bool = False):
        """Save extracted features for individual datasets (V422, ASAP, or ATEPP)
        Args:
            note_pairs: List of (score_notes, perf_params) tuples
            dataset_name: Name of the dataset ('v422', 'asap', 'atepp')
            use_midihum: Whether to use midihum features
        """
        midihum_suffix = "_midihum" if use_midihum else ""

        cache_file = os.path.join(self.config.output.artifacts_dir, 
                                f"{dataset_name}_features{midihum_suffix}_cache.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading {dataset_name} features, midihum: {use_midihum}, from cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"Loaded {dataset_name} features shape: {cache_data['context_features'].shape}, targets shape: {cache_data['targets'].shape} from cache")
            return cache_data['context_features'], cache_data['targets']

        print(f"Extracting features for {dataset_name} dataset...")
        training_notes, avg_tempos = [], []
        for idx, (score_notes, perf_params) in tqdm(enumerate(note_pairs)):
            if not self.config.train.enabled and idx == 1:
                print("Skipping further data loading for training, only using first piece")
                break
            
            # to investigate later!
            if score_notes.shape != perf_params.shape:
                # print(f"Warning: Score notes and performance parameters have different shapes: {score_notes.shape} != {perf_params.shape}")
                continue
            
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
        
        # Flatten all notes and extract features/targets
        all_notes = []
        for piece_notes in training_notes:
            all_notes.extend(piece_notes)
        
        # Filter notes with targets
        training_notes_filtered = [note for note in all_notes if 
                                 note.beat_period is not None and
                                 note.timing is not None and
                                 note.velocity is not None and
                                 note.articulation_log is not None]
        
        context_features = self.feature_extractor.encode_features(
            training_notes_filtered, 
            fit=True, 
            use_midihum=self.config.model.use_midihum_features
        )
        
        targets = np.array([[
            note.beat_period,
            note.timing,
            note.velocity,
            note.articulation_log
        ] for note in training_notes_filtered])
        
        # Save to cache
        cache_data = {
            'context_features': context_features,
            'targets': targets,
            'use_midihum': use_midihum,
            'dataset_name': dataset_name,
            'feature_shape': context_features.shape,
            'target_shape': targets.shape,
            'avg_tempos': avg_tempos
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Saved {dataset_name} features cache, midihum: {use_midihum}")
        print(f"Features shape: {context_features.shape}, Targets shape: {targets.shape}")
        return context_features, targets


    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        
        use_midihum = self.config.model.use_midihum_features
        
        print("Loading training data...")
        note_pairs = [] 
        context_features = []
        targets = []
        if self.use_vienna4x22: # default, most time will use 
            note_pairs = self.get_v422_aligned_note_arrays('train')
            context_features_vienna, targets_vienna = self.get_dataset_features(note_pairs, 'v422', use_midihum)
            context_features.append(context_features_vienna)
            targets.append(targets_vienna)
        elif self.asap_dir is not None:
            print("Loading ASAP training data...")
            note_pairs.extend(self.get_asap_aligned_note_arrays('train', self.asap_split_csv))
            context_features_asap, targets_asap = self.get_dataset_features(note_pairs, 'asap', use_midihum)
            context_features.append(context_features_asap)
            targets.append(targets_asap)
        if self.atepp_dir is not None:
            print("Loading ATEPP training data...")
            note_pairs.extend(self.get_atepp_aligned_note_arrays('train', self.atepp_meta_csv))
            context_features_atepp, targets_atepp = self.get_dataset_features(note_pairs, 'atepp', use_midihum)
            context_features.append(context_features_atepp)
            targets.append(targets_atepp)
        print(f"Loaded {len(note_pairs)} score-performance pairs")

        # Log dataset info
        # wandb.log({"dataset_size": len(note_pairs)})
        
        context_features = np.vstack(context_features)
        targets = np.vstack(targets)
        print(f"Total features shape: {context_features.shape}, Total targets shape: {targets.shape}")

        return context_features, targets
    
    def train(self):
        """Train the YQX system"""
        # Initialize wandb
        wandb.init(
            project="yqx-expressive-performance",
            config=dict(self.config),
            name=self._generate_model_identifier(self.config)
        )
        
        # Train model (model-agnostic interface)
        train_kwargs = {}
        if self.config.model.type == "bvae":
            bvae_config = self.config.model.bvae
            train_kwargs['epochs'] = bvae_config.get('epochs', 1000)
            train_kwargs['batch_size'] = bvae_config.get('batch_size', 32)
        elif self.config.model.type == "flow":
            flow_config = self.config.model.flow
            train_kwargs['epochs'] = flow_config.get('epochs', 100)
            train_kwargs['batch_size'] = flow_config.get('batch_size', 32)
        
        # filter out the data with nan in context_features 
        nan_mask = np.isnan(self.context_features).any(axis=1)
        self.context_features = self.context_features[~nan_mask]
        self.targets = self.targets[~nan_mask]

        self.model.train(self.context_features, self.targets, **train_kwargs)
        
        print(f"Training time: {time.time() - t0} seconds")
        
        # Save model and encoders
        self.save_model()
        
        # Finish wandb run
        wandb.finish()
    
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
        
        # Predict expressive parameters (model-agnostic)
        print("Predicting expressive parameters...")
        context_features = self.feature_extractor.encode_features(
            expressive_notes, 
            fit=False, 
            use_midihum=self.config.model.use_midihum_features
        )
        predictions = self.model.predict(context_features)
        
        # Convert predictions back to ExpressiveNote objects
        predicted_expressive_notes = []
        for i, note in enumerate(expressive_notes):
            new_note = ExpressiveNote(
                pitch=note.pitch,
                onset_beat=note.onset_beat,
                duration_beat=note.duration_beat,
                voice=note.voice,
                pitch_interval=note.pitch_interval,
                duration_ratio=note.duration_ratio,
                rhythmic_context=note.rhythmic_context,
                ir_label=note.ir_label,
                ir_closure=note.ir_closure,
                position_in_phrase=note.position_in_phrase,
                beat_period=float(np.clip(predictions[i, 0], 0.3, 3.0)),
                timing=float(np.clip(predictions[i, 1], -0.5, 0.5)),
                velocity=int(np.clip(predictions[i, 2] * 127, 1, 127)),
                articulation_log=float(np.clip(predictions[i, 3], -2.0, 1.0))
            )
            predicted_expressive_notes.append(new_note)
        
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
        """Save trained model with integrated scaler"""
        filepath = filepath or self.model_path
        print(f"Saving model to: {filepath}")
        self.model.save(filepath, self.feature_extractor.feature_scaler)
    
    def load_model(self, filepath: str = None):
        """Load trained model with integrated scaler"""
        filepath = filepath or self.model_path
        print(f"Loading model from: {filepath}")
        self.model.load(filepath)
        if self.model.feature_scaler is not None:
            self.feature_extractor.feature_scaler = self.model.feature_scaler

    def test_vienna4x22(self, output_subdir: str = None):
        """Test the model on all 4 Vienna4x22 pieces and organize outputs"""
        if not self.model.trained:
            print("Model not trained! Please train or load a model first.")
            return
        
        # Define test pieces with their designated tempos
        test_pieces = [
            ("Chopin_op10_no3", 30), 
            ("Chopin_op38", 120), 
            ("Mozart_K331_1st-mov", 120), 
            ("Schubert_D783_no15", 130)
        ]
        
        # Create organized output directory
        if output_subdir is None:
            # Generate subdir name from model identifier
            output_subdir = self._generate_model_identifier(self.config)
        
        test_output_dir = os.path.join(self.config.output.output_dir, output_subdir)
        os.makedirs(test_output_dir, exist_ok=True)
        
        print(f"Testing model on Vienna4x22 pieces...")
        print(f"Output directory: {test_output_dir}")
        
        results_summary = {
            "model_type": self.config.model.type,
            "model_config": dict(self.config.model),
            "model_path": self.model_path,
            "use_midihum_features": self.config.model.use_midihum_features,
            "pieces": {}
        }
        
        for piece_name, tempo in test_pieces:
            print(f"\nProcessing {piece_name} (tempo: {tempo} BPM)...")
            
            # Input and output paths
            input_score = os.path.join(self.musicxml_dir, f"{piece_name}.musicxml")
            output_midi = os.path.join(test_output_dir, f"{piece_name}_predicted.mid")
            
            # Render performance with piece-specific tempo
            self.render_performance(
                musicxml_path=input_score,
                output_midi=output_midi,
                initial_tempo=tempo
            )
            
            # Move plots to organized location
            pred_plot_src = os.path.join(self.config.output.artifacts_dir, "pred_params.png")
            pred_plot_dst = os.path.join(test_output_dir, f"{piece_name}_predictions.png")
            if os.path.exists(pred_plot_src):
                import shutil
                shutil.move(pred_plot_src, pred_plot_dst)
            
            results_summary["pieces"][piece_name] = {
                "input_score": input_score,
                "output_midi": output_midi,
                "predictions_plot": pred_plot_dst,
                "designated_tempo": tempo
            }
        
        # Save experiment summary
        import json
        summary_file = os.path.join(test_output_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n Test completed!")
        print(f"Results saved to: {test_output_dir}")
        print(f"Summary: {summary_file}")
        
        return test_output_dir

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
    
    if conf.test.enabled:
        print("Testing YQX system...")
        if not conf.train.enabled:
            yqx.load_model()
        
        print("Running Vienna4x22 test suite...")
        test_output_dir = yqx.test_vienna4x22()
    
    if conf.render.enabled:
        if not os.path.exists(conf.render.input_score):
            print(f"Error: Input score {conf.render.input_score} not found")
            return
            
        if not conf.train.enabled and not conf.test.enabled:
            yqx.load_model()

        print(f"Rendering performance of {conf.render.input_score}")
        yqx.render_performance(
            musicxml_path=conf.render.input_score,
            output_midi=conf.render.output_midi,
            initial_tempo=conf.render.initial_tempo
        )


if __name__ == "__main__":
    main()