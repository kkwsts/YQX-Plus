import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from expressivenote import *
from features import FeatureExtractor


class BaseConditioner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.projection = nn.Linear(input_dim, output_dim).to(device)

    def process(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class CategoricalFeatureConditioner(BaseConditioner):
    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        super().__init__(input_dim, output_dim, device)
        
    def tokenize(self, notes: List[ExpressiveNote]) -> torch.Tensor:
        features = []
        for note in notes:
            rhythmic_context = note.rhythmic_context or "boundary"
            ir_label = note.ir_label or "boundary"
            chord_character = note.chord_character or "none"
            chord_size = note.chord_size or "none"
            is_articulated = float(note.is_articulated) if note.is_articulated is not None else 0
            is_silent = float(note.is_silent) if note.is_silent is not None else 0
            phrase_position = note.phrase_position or "none"
            ir_type = note.ir_type or "none"
            
            rhythmic_idx = self._map_rhythmic_context(rhythmic_context)
            ir_label_idx = self._map_ir_label(ir_label)
            chord_char_idx = self._map_chord_character(chord_character)
            chord_size_idx = self._map_chord_size(chord_size)
            phrase_pos_idx = self._map_phrase_position(phrase_position)
            ir_type_idx = self._map_ir_type(ir_type)
            
            features.append([
                rhythmic_idx, ir_label_idx, chord_char_idx, chord_size_idx,
                is_articulated, is_silent, phrase_pos_idx, ir_type_idx
            ])
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _map_rhythmic_context(self, context: str) -> int:
        mapping = {"s-s-l": 0, "s-l-s": 1, "l-s-s": 2, "boundary": 3,
                   "s-l-l": 4, "l-l-s": 5, "l-s-l": 6}
        return mapping.get(context, 3)
    
    def _map_ir_label(self, label: str) -> int:
        mapping = {"Process": 0, "Reversal": 1, "Registral_Return": 2, 
                   "Intervallic_Duplication": 3, "boundary": 4}
        return mapping.get(label, 4)
    
    def _map_chord_character(self, character: str) -> int:
        mapping = {"major": 0, "minor": 1, "dominant": 2, "diminished": 3, 
                   "augmented": 4, "none": 5}
        return mapping.get(character, 5)
    
    def _map_chord_size(self, size: str) -> int:
        mapping = {"triad": 0, "seventh": 1, "fifth": 2, "none": 3}
        return mapping.get(size, 3)
    
    def _map_phrase_position(self, position: str) -> int:
        mapping = {"beginning": 0, "middle": 1, "end": 2, "none": 3}
        return mapping.get(position, 3)
    
    def _map_ir_type(self, ir_type: str) -> int:
        mapping = {"ascending": 0, "descending": 1, "static": 2, "none": 3}
        return mapping.get(ir_type, 3)


class ContinuousFeatureConditioner(BaseConditioner):
    def __init__(self, input_dim: int, output_dim: int, device: str = 'cpu'):
        super().__init__(input_dim, output_dim, device=device)
        
    def tokenize(self, notes: List[ExpressiveNote]) -> torch.Tensor:
        features = []
        for note in notes:
            pitch = note.pitch or 0
            onset_beat = note.onset_beat or 0
            duration_beat = note.duration_beat or 0
            pitch_interval = note.pitch_interval or 0
            duration_ratio = note.duration_ratio or 0
            ir_closure = note.ir_closure or 0
            position_in_phrase = note.position_in_phrase or 0
            
            pitch_class = note.pitch_class or 0
            octave = note.octave or 0
            follows_pause = float(note.follows_pause) if note.follows_pause is not None else 0
            time_since_last_pressed = note.time_since_last_pressed or 0
            time_since_last_released = note.time_since_last_released or 0
            
            features.append([
                pitch, onset_beat, duration_beat, pitch_interval,
                duration_ratio, ir_closure, position_in_phrase,
                pitch_class, octave, follows_pause,
                time_since_last_pressed, time_since_last_released,
            ])
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)


class MIDIHUMFeatureConditioner(BaseConditioner):
    def __init__(self, output_dim: int, device: str = 'cpu'):
        super().__init__(input_dim=54, output_dim=output_dim, device=device)
    
    def tokenize(self, notes: List[ExpressiveNote]) -> torch.Tensor:
        features = []
        for note in notes:
            pitch_class = note.pitch_class or 0
            octave = note.octave or 0
            follows_pause = float(note.follows_pause) if note.follows_pause is not None else 0
            chord_character = 0 if note.chord_character is None else self._map_chord(note.chord_character)
            chord_size = 0 if note.chord_size is None else self._map_chord_size(note.chord_size)
            
            time_feats = [
                note.time_since_last_pressed or 0,
                note.time_since_last_released or 0,
                note.time_since_pitch_class or 0,
                note.time_since_octave or 0,
                note.time_since_follows_pause or 0,
                note.time_since_chord_character or 0,
                note.time_since_chord_size or 0
            ]
            
            interval_feats = [
                note.interval_from_pressed or 0,
                note.interval_from_released or 0,
                note.abs_interval_from_pressed or 0,
                note.abs_interval_from_released or 0,
                note.log_abs_interval_from_pressed or 0,
                note.log_abs_interval_from_released or 0
            ]
            
            sma_feats = [
                getattr(note, f"pitch_sma_mean_15", 0),
                getattr(note, f"pitch_sma_min_15", 0),
                getattr(note, f"pitch_sma_max_15", 0),
                getattr(note, f"pitch_sma_std_15", 0),
                getattr(note, f"log_sustain_sma_mean_15", 0),
            ]
            
            tech_feats = [
                note.pitch_tenkan_sen or 0,
                note.pitch_kijun_sen or 0,
                note.pitch_senkou_span_a or 0,
                note.pitch_relative_to_tenkan_sen or 0,
                note.pitch_relative_to_kijun_sen or 0,
                note.pitch_relative_to_cloud or 0,
            ]
            
            feature_vec = [
                pitch_class, octave, follows_pause, chord_character, chord_size,
                *time_feats, *interval_feats, *sma_feats, *tech_feats
            ]
            
            if len(feature_vec) > 54:
                feature_vec = feature_vec[:54]
            elif len(feature_vec) < 54:
                feature_vec.extend([0] * (54 - len(feature_vec)))
            
            features.append(feature_vec)
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _map_chord(self, chord: Optional[str]) -> int:
        if chord is None: return -1
        mapping = {"major":0, "minor":1, "dominant":2, "diminished":3, "augmented":4, "none":5}
        return mapping.get(chord, -1)
    
    def _map_chord_size(self, size: Optional[str]) -> int:
        if size is None: return -1
        mapping = {"triad":0, "seventh":1, "fifth":2, "none":3}
        return mapping.get(size, -1)


class ExpressiveConditioningProvider:
    def __init__(self, 
                 categorical_dim: int = 8,  
                 continuous_dim: int = 54,
                 use_midihum: bool = False,
                 embed_dim: int = 128, 
                 device: str = 'cpu'):
        self.device = device
        self.use_midihum = use_midihum
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim
        
        self.categorical_conditioner = CategoricalFeatureConditioner(categorical_dim, embed_dim, device)
        self.continuous_conditioner = ContinuousFeatureConditioner(continuous_dim, embed_dim, device)
        self.midihum_conditioner = MIDIHUMFeatureConditioner(embed_dim, device) if use_midihum else None        
        
        self.processors = {
            'categorical': self.categorical_conditioner,
            'continuous': self.continuous_conditioner,
            'midihum': self.midihum_conditioner
        }
    
    def process(self, attributes: ConditioningAttributes, feature_extractor: FeatureExtractor = None) -> Dict[str, torch.Tensor]:
        notes = attributes.expressive_notes

        if feature_extractor:
            encoded_features = feature_extractor.encode_features(notes, fit=False, use_midihum=self.use_midihum)
            if not isinstance(encoded_features, np.ndarray):
                encoded_features = np.array(encoded_features)
            
            categorical_features = encoded_features[:, :self.categorical_dim]
            continuous_features = encoded_features[:, self.categorical_dim:]
            
            categorical_tensor = torch.tensor(categorical_features, dtype=torch.float32, device=self.device)
            continuous_tensor = torch.tensor(continuous_features, dtype=torch.float32, device=self.device)
            
            processed = {
                'categorical': self.categorical_conditioner.process(categorical_tensor),
                'continuous': self.continuous_conditioner.process(continuous_tensor)
            }
            
            if self.use_midihum and self.midihum_conditioner:
                midihum_features = continuous_tensor[:, -54:]  # 根据实际midihum特征数量调整
                processed['midihum'] = self.midihum_conditioner.process(midihum_features)
            else:
                batch_size = len(notes)
                default_midihum = torch.zeros(batch_size, 54, device=self.device)
                processed['midihum'] = default_midihum
        else:
            processed = {
                'categorical': self.categorical_conditioner.process(self.categorical_conditioner.tokenize(notes)),
                'continuous': self.continuous_conditioner.process(self.continuous_conditioner.tokenize(notes))
            }
            
            if self.use_midihum and self.midihum_conditioner:
                midihum_features = self.midihum_conditioner.tokenize(notes)
                processed['midihum'] = self.midihum_conditioner.process(midihum_features)
            else:
                batch_size = len(notes)
                default_midihum = torch.zeros(batch_size, 54, device=self.device)
                processed['midihum'] = default_midihum
                
        return processed