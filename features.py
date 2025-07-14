import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from expressivenote import ExpressiveNote
from midihum_chord_identifier import chord_attributes


class FeatureExtractor:
    """Extract musical context features from note arrays"""
    
    def __init__(self):
        self.ir_categories = [
            'Process', 'Reversal', 'Registral_Return', 'Intervallic_Duplication'
        ]
        
        # Fixed mappings - no need to save/load these
        self.feature_encoders = {
            'rhythmic_context': {
                'boundary': 0,
                's-s': 1, 's-m': 2, 's-l': 3,
                'm-s': 4, 'm-m': 5, 'm-l': 6,
                'l-s': 7, 'l-m': 8, 'l-l': 9
            },
            'ir_label': {
                'boundary': 0,
                'Process': 1,
                'Reversal': 2,
                'Registral_Return': 3,
                'Intervallic_Duplication': 4
            }
        }
        
        # Add midihum categorical mappings if needed
        if True:  # Always define these for consistency
            self.feature_encoders.update({
                'chord_character_pressed': {
                    'none': 0, 'major': 1, 'minor': 2, 'diminished': 3, 'augmented': 4,
                    'dominant': 5, 'major7': 6, 'minor7': 7, 'other': 8
                },
                'chord_size_pressed': {
                    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8
                },
                'chord_character': {
                    'none': 0, 'major': 1, 'minor': 2, 'diminished': 3, 'augmented': 4,
                    'dominant': 5, 'major7': 6, 'minor7': 7, 'other': 8
                },
                'chord_size': {
                    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8
                }
            })
        
        self.feature_scaler = None
    
    def encode_features(self, notes: List[ExpressiveNote], fit: bool = False, use_midihum: bool = False) -> np.ndarray:
        """Encode all features (both categorical and continuous) to numerical arrays
        
        Args:
            notes: List of ExpressiveNote objects
            fit: Whether to fit the encoders and scaler (True for training, False for inference)
            use_midihum: Whether to include additional midihum features
            
        Returns:
            Numpy array of encoded features
        """
        categorical_features = []
        continuous_features = []
        
        # Define base categorical features
        BASE_CATEGORICAL_FEATURES = [
            'rhythmic_context',
            'ir_label'
        ]
        
        # Define midihum categorical features
        MIDIHUM_CATEGORICAL_FEATURES = [
            'chord_character_pressed',
            'chord_size_pressed',
            'chord_character',
            'chord_size'
        ]
        
        # Define base continuous features
        BASE_FEATURES = [
            'pitch',
            'onset_beat',
            'duration_beat',
            'pitch_interval',
            'duration_ratio',
            'ir_closure',
            'position_in_phrase'
        ]
        
        # Define midihum basic features
        MIDIHUM_BASIC_FEATURES = [
            'pitch_class',
            'octave'
        ]
        
        # Define midihum time features
        MIDIHUM_TIME_FEATURES = [
            'time_since_last_pressed',
            'time_since_last_released',
            'time_since_pitch_class',
            'time_since_octave',
            'time_since_follows_pause',
            'time_since_chord_character',
            'time_since_chord_size'
        ]
        
        LOG_TIME_SINCE_FEATURES = [f'log_{f}' for f in MIDIHUM_TIME_FEATURES]
        
        INTERVAL_FEATURES = [
            'interval_from_pressed',
            'interval_from_released',
            'abs_interval_from_pressed',
            'abs_interval_from_released',
            'log_abs_interval_from_pressed',
            'log_abs_interval_from_released'
        ]
        
        # Technical indicator base features
        TECH_BASE_FEATURES = [
            'tenkan_sen',
            'kijun_sen',
            'senkou_span_a',
            'senkou_span_b',
            'chikou_span',
            'cloud_is_green',
            'relative_to_tenkan_sen',
            'relative_to_kijun_sen',
            'tenkan_sen_relative_to_kijun_sen',
            'relative_to_chikou_span',
            'relative_to_cloud'
        ]
        
        # Technical indicator target variables
        TECH_TARGETS = [
            'pitch',
            'log_sustain',
            'interval_from_released',
            'interval_from_pressed'
        ]
        
        for note in notes:
            # Base categorical features
            cat_features = [getattr(note, f) for f in BASE_CATEGORICAL_FEATURES]
            
            if use_midihum:
                cat_features.extend([getattr(note, f) for f in MIDIHUM_CATEGORICAL_FEATURES])
            
            categorical_features.append(cat_features)
            
            # Continuous features
            cont_features = []
            
            cont_features.extend([getattr(note, f) for f in BASE_FEATURES])
            
            if use_midihum:
                cont_features.append([getattr(note, f) for f in MIDIHUM_BASIC_FEATURES])
                cont_features.append(float(note.follows_pause if note.follows_pause is not None else 0))
                cont_features.append(note.num_played_notes_pressed)
                cont_features.append(note.avg_pitch_pressed)
                
                # Add time since features
                cont_features.extend([getattr(note, f) for f in MIDIHUM_TIME_FEATURES])
                
                # Add log time since features
                cont_features.extend([getattr(note, f) for f in LOG_TIME_SINCE_FEATURES])
                
                # Add interval features
                cont_features.extend([getattr(note, f) for f in INTERVAL_FEATURES])
                
                # Add moving average features for pitch and log_sustain
                for target in ['pitch', 'log_sustain']:
                    for window in [15, 30, 75]:
                        # Basic SMA features
                        sma_types = ['mean', 'min', 'max', 'std']
                        for sma_type in sma_types:
                            # Regular SMA
                            cont_features.append(getattr(note, f'{target}_sma_{sma_type}_{window}'))
                            # Forward SMA
                            cont_features.append(getattr(note, f'{target}_fwd_sma_{sma_type}_{window}'))
                            # Regular SMA oscillator
                            cont_features.append(getattr(note, f'{target}_sma_{sma_type}_{window}_oscillator'))
                            # Forward SMA oscillator
                            cont_features.append(getattr(note, f'{target}_fwd_sma_{sma_type}_{window}_oscillator'))
                
                # Add technical indicators
                for target in TECH_TARGETS:
                    for feature in TECH_BASE_FEATURES:
                        feature_name = f'{target}_{feature}'
                        cont_features.append(getattr(note, feature_name))
            
            # Replace None values with 0.0
            continuous_features.append([x if x is not None else 0.0 for x in cont_features])
        

        # Apply encodings
        encoded_categorical = []
        for features in categorical_features:
            encoded = []
            for i, (feature, name) in enumerate(zip(features, self.feature_encoders.keys())):
                encoded.append(self.feature_encoders[name].get(feature, 0))
            encoded_categorical.append(encoded)
        
        # Combine and scale features
        all_features = np.hstack([np.array(encoded_categorical), np.array(continuous_features)])
        
        if fit:
            self.feature_scaler = StandardScaler()
            all_features = self.feature_scaler.fit_transform(all_features)
        elif self.feature_scaler is not None:
            all_features = self.feature_scaler.transform(all_features)
        
        return all_features
    
    def extract_voices(self, note_array: np.ndarray) -> List[np.ndarray]:
        """Extract all voices sorted from highest to lowest pitch"""
        # Group notes by voice
        unique_voices = np.unique(note_array['voice'])
        voice_groups = {}
        
        # Collect notes for each voice
        for voice in unique_voices:
            voice_notes = note_array[note_array['voice'] == voice]
            if len(voice_notes) > 0:
                # Calculate average pitch for this voice
                avg_pitch = np.mean(voice_notes['pitch'])
                voice_groups[voice] = {
                    'notes': voice_notes,
                    'avg_pitch': avg_pitch
                }
        
        # Sort voices by average pitch (highest to lowest)
        if len(voice_groups) > 0:
            sorted_voices = sorted(voice_groups.keys(),
                                 key=lambda v: voice_groups[v]['avg_pitch'],
                                 reverse=True)
            
            # Create list of sorted voice note arrays
            voice_note_arrays = []
            for voice in sorted_voices:
                voice_notes = voice_groups[voice]['notes']
                # Sort notes within each voice by onset beat
                sort_idx = np.argsort(voice_notes['onset_beat'])
                voice_notes = voice_notes[sort_idx]
                voice_note_arrays.append(voice_notes)
                
            return voice_note_arrays
        else:
            return [np.array([], dtype=note_array.dtype)]
    
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
        """Implication-Realization analysis based on Narmour's principles"""
        if idx == 0 or idx >= len(pitches) - 2:
            return "boundary", 0.0
        
        # Analyze melodic intervals
        int1 = pitches[idx] - pitches[idx-1]  # Implicative interval
        int2 = pitches[idx+1] - pitches[idx]  # Realized interval
        
        # Check registral direction
        same_direction = (int1 * int2) > 0
        
        # Analyze based on IR principles
        if abs(int1) <= 5:  # Small implicative interval
            # Registral direction principle for small intervals
            if same_direction and abs(abs(int1) - abs(int2)) <= 3:
                ir_label = "Process"  # Good continuation
                closure = -0.5  # Low closure (continuing)
            elif not same_direction and abs(abs(int1) - abs(int2)) <= 2:
                ir_label = "Intervallic_Duplication"
                closure = 0.2  # Some closure due to direction change
            else:
                ir_label = "Reversal"
                closure = 0.4  # Medium closure
        else:  # Large implicative interval
            # Check for registral return
            return_interval = pitches[idx+1] - pitches[idx-1]
            if abs(return_interval) <= 2:  # Within 2 semitones of first note
                ir_label = "Registral_Return"
                closure = 0.8  # High closure
            elif not same_direction and abs(int2) < abs(int1) - 3:
                ir_label = "Reversal"  # Large interval implies direction change
                closure = 0.6  # Substantial closure
            else:
                ir_label = "Process"
                closure = 0.3  # Some closure
        
        # Additional closure based on proximity
        if abs(int2) <= 5:  # Small realized interval adds stability
            closure -= 0.1
        
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
    
    def extract_features(self, score_notes: np.ndarray, parameters: Optional[np.ndarray] = None, 
                         plot: Optional[bool] = False, use_midihum_features: bool = False) -> List[ExpressiveNote]:
        """
        Extract comprehensive features from score and parameters (if available).
        Features are organized into pitch, voice, rhythm, and phrase categories.
        If use_midihum_features is True, augment features using MidiHumFeatureEngineer.
        """
        voices = self.extract_voices(score_notes)
        phrase_boundaries = self.detect_phrase_boundaries(score_notes)
        
        # Pre-compute global statistics for normalization
        all_pitches = score_notes['pitch']
        all_durations = score_notes['duration_beat']
        piece_pitch_range = (all_pitches.min(), all_pitches.max())
        piece_avg_duration = all_durations.mean()
        
        # Pre-compute voice statistics
        voice_stats = {}
        for voice_idx, voice_notes in enumerate(voices):
            voice_pitches = voice_notes['pitch']
            voice_durations = voice_notes['duration_beat']
            voice_stats[voice_idx] = {
                'pitch_range': (voice_pitches.min(), voice_pitches.max()),
                'avg_duration': voice_durations.mean(),
                'duration_ranks': np.argsort(np.argsort(voice_durations)) / len(voice_durations)
            }
        
        # Pre-compute duration ranks for the entire piece
        piece_duration_ranks = np.argsort(np.argsort(all_durations)) / len(all_durations)

        expressive_notes = []
        
        for voice_idx, voice_notes in enumerate(voices):
            # Compute voice layer information
            voice_pitches = voice_notes['pitch']
            voice_layers = self._compute_voice_layers(voice_pitches)
            
            for i, note in enumerate(voice_notes):
                # Basic note information
                pitch = note['pitch']
                onset_beat = note['onset_beat']
                duration_beat = note['duration_beat']
                voice = note['voice']
                
                # ========================================
                # PITCH FEATURES
                # ========================================
                pitch_features = self._extract_pitch_features(
                    voice_notes, i, pitch, piece_pitch_range, voice_stats[voice_idx]
                )
                
                # ========================================
                # VOICE FEATURES  
                # ========================================
                voice_features = self._extract_voice_features(
                    score_notes, voice_notes, i, pitch, onset_beat, voice_layers[i]
                )
                
                # ========================================
                # RHYTHMIC FEATURES
                # ========================================
                rhythmic_features = self._extract_rhythmic_features(
                    voice_notes, i, duration_beat, piece_avg_duration, 
                    voice_stats[voice_idx], piece_duration_ranks
                )
                
                # ========================================
                # PHRASE FEATURES
                # ========================================
                phrase_features = self._extract_phrase_features(
                    voice_notes, i, phrase_boundaries
                )
                
                # ========================================
                # EXPRESSIVE TARGETS
                # ========================================
                beat_period = None
                timing = None
                velocity = None
                articulation_log = None
                
                if parameters is not None and i < len(parameters):
                    perf_param = parameters[i]
                    beat_period = perf_param['beat_period'] 
                    timing = perf_param['timing'] 
                    velocity = perf_param['velocity']
                    articulation_log = perf_param['articulation_log'] 
                
                # Create ExpressiveNote with all features
                expressive_note = ExpressiveNote(
                    # Basic features
                    pitch=pitch,
                    onset_beat=onset_beat,
                    duration_beat=duration_beat,
                    voice=voice,
                    
                    # Pitch features
                    **pitch_features,
                    
                    # Voice features
                    **voice_features,
                    
                    # Rhythmic features
                    **rhythmic_features,
                    
                    # Phrase features
                    **phrase_features,
                    
                    # Expressive targets
                    beat_period=beat_period,
                    timing=timing,
                    velocity=velocity,
                    articulation_log=articulation_log
                )
                
                expressive_notes.append(expressive_note)

        if use_midihum_features:
            # Convert ExpressiveNote list to DataFrame
            from dataclasses import asdict
            note_df = pd.DataFrame([asdict(n) for n in expressive_notes])
            # Use the original note array for midihum features (for chord context, etc.)
            midihum_engineer = MidiHumFeatureEngineer()
            midihum_df = midihum_engineer.add_midihum_features(score_notes)
            # Merge midihum features into note_df (align by onset_beat and pitch)
            merged = pd.merge(
                note_df,
                midihum_df,
                left_on=['onset_beat', 'pitch'],
                right_on=['time', 'pitch'],
                how='left',
                suffixes=('', '_midihum')
            )
            
            # Map midihum features to ExpressiveNote fields
            expressive_notes = []
            for _, row in merged.iterrows():
                note = ExpressiveNote(
                    # Original YQX features
                    pitch=row['pitch'],
                    onset_beat=row['onset_beat'],
                    duration_beat=row['duration_beat'],
                    voice=row['voice'],
                    pitch_interval=row['pitch_interval'],
                    duration_ratio=row['duration_ratio'],
                    rhythmic_context=row['rhythmic_context'],
                    ir_label=row['ir_label'],
                    ir_closure=row['ir_closure'],
                    position_in_phrase=row['position_in_phrase'],
                    beat_period=row['beat_period'],
                    timing=row['timing'],
                    velocity=row['velocity'],
                    articulation_log=row['articulation_log'],
                    
                    # Basic midihum features
                    pitch_class=row['pitch_class'],
                    octave=row['octave'],
                    follows_pause=bool(row['follows_pause']),
                    
                    # Chord context features
                    chord_character_pressed=row['chord_character_pressed'],
                    chord_size_pressed=row['chord_size_pressed'],
                    chord_character=row['chord_character'],
                    chord_size=row['chord_size'],
                    num_played_notes_pressed=row['num_played_notes_pressed'],
                    avg_pitch_pressed=row['avg_pitch_pressed'],
                    
                    # Timing context features
                    time_since_last_pressed=row['time_since_last_pressed'],
                    time_since_last_released=row['time_since_last_released'],
                    
                    # Time since various events
                    time_since_pitch_class=row['time_since_pitch_class'],
                    time_since_octave=row['time_since_octave'],
                    time_since_follows_pause=row['time_since_follows_pause'],
                    time_since_chord_character=row['time_since_chord_character'],
                    time_since_chord_size=row['time_since_chord_size'],
                    
                    # Log versions of time features
                    log_time_since_pitch_class=row['log_time_since_pitch_class'],
                    log_time_since_octave=row['log_time_since_octave'],
                    log_time_since_follows_pause=row['log_time_since_follows_pause'],
                    log_time_since_chord_character=row['log_time_since_chord_character'],
                    log_time_since_chord_size=row['log_time_since_chord_size'],
                    log_time_since_last_pressed=row['log_time_since_last_pressed'],
                    log_time_since_last_released=row['log_time_since_last_released'],
                    
                    # Interval features
                    interval_from_pressed=row['interval_from_pressed'],
                    interval_from_released=row['interval_from_released'],
                    abs_interval_from_pressed=row['abs_interval_from_pressed'],
                    abs_interval_from_released=row['abs_interval_from_released'],
                    log_abs_interval_from_pressed=row['log_abs_interval_from_pressed'],
                    log_abs_interval_from_released=row['log_abs_interval_from_released'],
                    
                    # Moving average features for pitch (all windows)
                    pitch_sma_mean_15=row['pitch_sma_mean_15'],
                    pitch_sma_min_15=row['pitch_sma_min_15'],
                    pitch_sma_max_15=row['pitch_sma_max_15'],
                    pitch_sma_std_15=row['pitch_sma_std_15'],
                    pitch_sma_mean_30=row['pitch_sma_mean_30'],
                    pitch_sma_min_30=row['pitch_sma_min_30'],
                    pitch_sma_max_30=row['pitch_sma_max_30'],
                    pitch_sma_std_30=row['pitch_sma_std_30'],
                    pitch_sma_mean_75=row['pitch_sma_mean_75'],
                    pitch_sma_min_75=row['pitch_sma_min_75'],
                    pitch_sma_max_75=row['pitch_sma_max_75'],
                    pitch_sma_std_75=row['pitch_sma_std_75'],
                    
                    # Forward pitch SMA
                    pitch_fwd_sma_mean_15=row['pitch_fwd_sma_mean_15'],
                    pitch_fwd_sma_min_15=row['pitch_fwd_sma_min_15'],
                    pitch_fwd_sma_max_15=row['pitch_fwd_sma_max_15'],
                    pitch_fwd_sma_std_15=row['pitch_fwd_sma_std_15'],
                    pitch_fwd_sma_mean_30=row['pitch_fwd_sma_mean_30'],
                    pitch_fwd_sma_min_30=row['pitch_fwd_sma_min_30'],
                    pitch_fwd_sma_max_30=row['pitch_fwd_sma_max_30'],
                    pitch_fwd_sma_std_30=row['pitch_fwd_sma_std_30'],
                    pitch_fwd_sma_mean_75=row['pitch_fwd_sma_mean_75'],
                    pitch_fwd_sma_min_75=row['pitch_fwd_sma_min_75'],
                    pitch_fwd_sma_max_75=row['pitch_fwd_sma_max_75'],
                    pitch_fwd_sma_std_75=row['pitch_fwd_sma_std_75'],
                    
                    # Pitch SMA oscillators
                    pitch_sma_mean_15_oscillator=row['pitch_sma_mean_15_oscillator'],
                    pitch_sma_min_15_oscillator=row['pitch_sma_min_15_oscillator'],
                    pitch_sma_max_15_oscillator=row['pitch_sma_max_15_oscillator'],
                    pitch_sma_std_15_oscillator=row['pitch_sma_std_15_oscillator'],
                    pitch_sma_mean_30_oscillator=row['pitch_sma_mean_30_oscillator'],
                    pitch_sma_min_30_oscillator=row['pitch_sma_min_30_oscillator'],
                    pitch_sma_max_30_oscillator=row['pitch_sma_max_30_oscillator'],
                    pitch_sma_std_30_oscillator=row['pitch_sma_std_30_oscillator'],
                    pitch_sma_mean_75_oscillator=row['pitch_sma_mean_75_oscillator'],
                    pitch_sma_min_75_oscillator=row['pitch_sma_min_75_oscillator'],
                    pitch_sma_max_75_oscillator=row['pitch_sma_max_75_oscillator'],
                    pitch_sma_std_75_oscillator=row['pitch_sma_std_75_oscillator'],
                    
                    # Forward pitch SMA oscillators
                    pitch_fwd_sma_mean_15_oscillator=row['pitch_fwd_sma_mean_15_oscillator'],
                    pitch_fwd_sma_min_15_oscillator=row['pitch_fwd_sma_min_15_oscillator'],
                    pitch_fwd_sma_max_15_oscillator=row['pitch_fwd_sma_max_15_oscillator'],
                    pitch_fwd_sma_std_15_oscillator=row['pitch_fwd_sma_std_15_oscillator'],
                    pitch_fwd_sma_mean_30_oscillator=row['pitch_fwd_sma_mean_30_oscillator'],
                    pitch_fwd_sma_min_30_oscillator=row['pitch_fwd_sma_min_30_oscillator'],
                    pitch_fwd_sma_max_30_oscillator=row['pitch_fwd_sma_max_30_oscillator'],
                    pitch_fwd_sma_std_30_oscillator=row['pitch_fwd_sma_std_30_oscillator'],
                    pitch_fwd_sma_mean_75_oscillator=row['pitch_fwd_sma_mean_75_oscillator'],
                    pitch_fwd_sma_min_75_oscillator=row['pitch_fwd_sma_min_75_oscillator'],
                    pitch_fwd_sma_max_75_oscillator=row['pitch_fwd_sma_max_75_oscillator'],
                    pitch_fwd_sma_std_75_oscillator=row['pitch_fwd_sma_std_75_oscillator'],
                    
                    # Log sustain SMA features
                    log_sustain_sma_mean_15=row['log_sustain_sma_mean_15'],
                    log_sustain_sma_min_15=row['log_sustain_sma_min_15'],
                    log_sustain_sma_max_15=row['log_sustain_sma_max_15'],
                    log_sustain_sma_std_15=row['log_sustain_sma_std_15'],
                    log_sustain_sma_mean_30=row['log_sustain_sma_mean_30'],
                    log_sustain_sma_min_30=row['log_sustain_sma_min_30'],
                    log_sustain_sma_max_30=row['log_sustain_sma_max_30'],
                    log_sustain_sma_std_30=row['log_sustain_sma_std_30'],
                    log_sustain_sma_mean_75=row['log_sustain_sma_mean_75'],
                    log_sustain_sma_min_75=row['log_sustain_sma_min_75'],
                    log_sustain_sma_max_75=row['log_sustain_sma_max_75'],
                    log_sustain_sma_std_75=row['log_sustain_sma_std_75'],
                    
                    # Forward log sustain SMA
                    log_sustain_fwd_sma_mean_15=row['log_sustain_fwd_sma_mean_15'],
                    log_sustain_fwd_sma_min_15=row['log_sustain_fwd_sma_min_15'],
                    log_sustain_fwd_sma_max_15=row['log_sustain_fwd_sma_max_15'],
                    log_sustain_fwd_sma_std_15=row['log_sustain_fwd_sma_std_15'],
                    log_sustain_fwd_sma_mean_30=row['log_sustain_fwd_sma_mean_30'],
                    log_sustain_fwd_sma_min_30=row['log_sustain_fwd_sma_min_30'],
                    log_sustain_fwd_sma_max_30=row['log_sustain_fwd_sma_max_30'],
                    log_sustain_fwd_sma_std_30=row['log_sustain_fwd_sma_std_30'],
                    log_sustain_fwd_sma_mean_75=row['log_sustain_fwd_sma_mean_75'],
                    log_sustain_fwd_sma_min_75=row['log_sustain_fwd_sma_min_75'],
                    log_sustain_fwd_sma_max_75=row['log_sustain_fwd_sma_max_75'],
                    log_sustain_fwd_sma_std_75=row['log_sustain_fwd_sma_std_75'],
                    
                    # Log sustain SMA oscillators
                    log_sustain_sma_mean_15_oscillator=row['log_sustain_sma_mean_15_oscillator'],
                    log_sustain_sma_min_15_oscillator=row['log_sustain_sma_min_15_oscillator'],
                    log_sustain_sma_max_15_oscillator=row['log_sustain_sma_max_15_oscillator'],
                    log_sustain_sma_std_15_oscillator=row['log_sustain_sma_std_15_oscillator'],
                    log_sustain_sma_mean_30_oscillator=row['log_sustain_sma_mean_30_oscillator'],
                    log_sustain_sma_min_30_oscillator=row['log_sustain_sma_min_30_oscillator'],
                    log_sustain_sma_max_30_oscillator=row['log_sustain_sma_max_30_oscillator'],
                    log_sustain_sma_std_30_oscillator=row['log_sustain_sma_std_30_oscillator'],
                    log_sustain_sma_mean_75_oscillator=row['log_sustain_sma_mean_75_oscillator'],
                    log_sustain_sma_min_75_oscillator=row['log_sustain_sma_min_75_oscillator'],
                    log_sustain_sma_max_75_oscillator=row['log_sustain_sma_max_75_oscillator'],
                    log_sustain_sma_std_75_oscillator=row['log_sustain_sma_std_75_oscillator'],
                    
                    # Forward log sustain SMA oscillators
                    log_sustain_fwd_sma_mean_15_oscillator=row['log_sustain_fwd_sma_mean_15_oscillator'],
                    log_sustain_fwd_sma_min_15_oscillator=row['log_sustain_fwd_sma_min_15_oscillator'],
                    log_sustain_fwd_sma_max_15_oscillator=row['log_sustain_fwd_sma_max_15_oscillator'],
                    log_sustain_fwd_sma_std_15_oscillator=row['log_sustain_fwd_sma_std_15_oscillator'],
                    log_sustain_fwd_sma_mean_30_oscillator=row['log_sustain_fwd_sma_mean_30_oscillator'],
                    log_sustain_fwd_sma_min_30_oscillator=row['log_sustain_fwd_sma_min_30_oscillator'],
                    log_sustain_fwd_sma_max_30_oscillator=row['log_sustain_fwd_sma_max_30_oscillator'],
                    log_sustain_fwd_sma_std_30_oscillator=row['log_sustain_fwd_sma_std_30_oscillator'],
                    log_sustain_fwd_sma_mean_75_oscillator=row['log_sustain_fwd_sma_mean_75_oscillator'],
                    log_sustain_fwd_sma_min_75_oscillator=row['log_sustain_fwd_sma_min_75_oscillator'],
                    log_sustain_fwd_sma_max_75_oscillator=row['log_sustain_fwd_sma_max_75_oscillator'],
                    log_sustain_fwd_sma_std_75_oscillator=row['log_sustain_fwd_sma_std_75_oscillator'],
                    
                    # Technical indicators for pitch
                    pitch_tenkan_sen=row['pitch_tenkan_sen'],
                    pitch_kijun_sen=row['pitch_kijun_sen'],
                    pitch_senkou_span_a=row['pitch_senkou_span_a'],
                    pitch_senkou_span_b=row['pitch_senkou_span_b'],
                    pitch_chikou_span=row['pitch_chikou_span'],
                    pitch_cloud_is_green=row['pitch_cloud_is_green'],
                    pitch_relative_to_tenkan_sen=row['pitch_relative_to_tenkan_sen'],
                    pitch_relative_to_kijun_sen=row['pitch_relative_to_kijun_sen'],
                    pitch_tenkan_sen_relative_to_kijun_sen=row['pitch_tenkan_sen_relative_to_kijun_sen'],
                    pitch_relative_to_chikou_span=row['pitch_relative_to_chikou_span'],
                    pitch_relative_to_cloud=row['pitch_relative_to_cloud'],
                    
                    # Technical indicators for log_sustain
                    log_sustain_tenkan_sen=row['log_sustain_tenkan_sen'],
                    log_sustain_kijun_sen=row['log_sustain_kijun_sen'],
                    log_sustain_senkou_span_a=row['log_sustain_senkou_span_a'],
                    log_sustain_senkou_span_b=row['log_sustain_senkou_span_b'],
                    log_sustain_chikou_span=row['log_sustain_chikou_span'],
                    log_sustain_cloud_is_green=row['log_sustain_cloud_is_green'],
                    log_sustain_relative_to_tenkan_sen=row['log_sustain_relative_to_tenkan_sen'],
                    log_sustain_relative_to_kijun_sen=row['log_sustain_relative_to_kijun_sen'],
                    log_sustain_tenkan_sen_relative_to_kijun_sen=row['log_sustain_tenkan_sen_relative_to_kijun_sen'],
                    log_sustain_relative_to_chikou_span=row['log_sustain_relative_to_chikou_span'],
                    log_sustain_relative_to_cloud=row['log_sustain_relative_to_cloud'],
                    
                    # Technical indicators for interval_from_released
                    interval_from_released_tenkan_sen=row['interval_from_released_tenkan_sen'],
                    interval_from_released_kijun_sen=row['interval_from_released_kijun_sen'],
                    interval_from_released_senkou_span_a=row['interval_from_released_senkou_span_a'],
                    interval_from_released_senkou_span_b=row['interval_from_released_senkou_span_b'],
                    interval_from_released_chikou_span=row['interval_from_released_chikou_span'],
                    interval_from_released_cloud_is_green=row['interval_from_released_cloud_is_green'],
                    interval_from_released_relative_to_tenkan_sen=row['interval_from_released_relative_to_tenkan_sen'],
                    interval_from_released_relative_to_kijun_sen=row['interval_from_released_relative_to_kijun_sen'],
                    interval_from_released_tenkan_sen_relative_to_kijun_sen=row['interval_from_released_tenkan_sen_relative_to_kijun_sen'],
                    interval_from_released_relative_to_chikou_span=row['interval_from_released_relative_to_chikou_span'],
                    interval_from_released_relative_to_cloud=row['interval_from_released_relative_to_cloud'],
                    
                    # Technical indicators for interval_from_pressed
                    interval_from_pressed_tenkan_sen=row['interval_from_pressed_tenkan_sen'],
                    interval_from_pressed_kijun_sen=row['interval_from_pressed_kijun_sen'],
                    interval_from_pressed_senkou_span_a=row['interval_from_pressed_senkou_span_a'],
                    interval_from_pressed_senkou_span_b=row['interval_from_pressed_senkou_span_b'],
                    interval_from_pressed_chikou_span=row['interval_from_pressed_chikou_span'],
                    interval_from_pressed_cloud_is_green=row['interval_from_pressed_cloud_is_green'],
                    interval_from_pressed_relative_to_tenkan_sen=row['interval_from_pressed_relative_to_tenkan_sen'],
                    interval_from_pressed_relative_to_kijun_sen=row['interval_from_pressed_relative_to_kijun_sen'],
                    interval_from_pressed_tenkan_sen_relative_to_kijun_sen=row['interval_from_pressed_tenkan_sen_relative_to_kijun_sen'],
                    interval_from_pressed_relative_to_chikou_span=row['interval_from_pressed_relative_to_chikou_span'],
                    interval_from_pressed_relative_to_cloud=row['interval_from_pressed_relative_to_cloud']
                )
                expressive_notes.append(note)

        return expressive_notes


    def plot_targets(self, expressive_notes, save_path):
        """Plot expressive parameters for visualization
        
        Args:
            expressive_notes: List of ExpressiveNote objects containing expressive parameters
        """
        import matplotlib.pyplot as plt
        
        # Group notes by voice
        voice_groups = {}
        for note in expressive_notes:
            if note.voice not in voice_groups:
                voice_groups[note.voice] = {
                    'onsets': [],
                    'beat_periods': [],
                    'timings': [],
                    'velocities': [],
                    'articulations': []
                }
            voice_groups[note.voice]['onsets'].append(note.onset_beat)
            voice_groups[note.voice]['beat_periods'].append(note.beat_period)
            voice_groups[note.voice]['timings'].append(note.timing)
            voice_groups[note.voice]['velocities'].append(note.velocity)
            voice_groups[note.voice]['articulations'].append(note.articulation_log)

        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Expressive Parameters')

        # Colors for different voices
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']

        # Plot each voice in each subplot
        for i, (voice, data) in enumerate(voice_groups.items()):
            color = colors[i % len(colors)]
            label = f'Voice {voice}'

            # Plot beat period
            ax1.plot(data['onsets'], data['beat_periods'], f'{color}.-', label=label, alpha=0.5)
            ax1.set_xlabel('Score Time (beats)')
            ax1.set_ylabel('Beat Period (s)')
            ax1.set_title('Tempo')
            ax1.grid(True)
            ax1.legend()

            # Plot timing
            ax2.plot(data['onsets'], data['timings'], f'{color}.-', label=label, alpha=0.5)
            ax2.set_xlabel('Score Time (beats)')
            ax2.set_ylabel('Timing Deviation (s)')
            ax2.set_title('Timing')
            ax2.grid(True)
            ax2.legend()

            # Plot velocity
            ax3.plot(data['onsets'], data['velocities'], f'{color}.-', label=label, alpha=0.5)
            ax3.set_xlabel('Score Time (beats)')
            ax3.set_ylabel('Velocity (0-127)')
            ax3.set_title('Dynamics')
            ax3.grid(True)
            ax3.legend()

            # Plot articulation
            ax4.plot(data['onsets'], data['articulations'], f'{color}.-', label=label, alpha=0.5)
            ax4.set_xlabel('Score Time (beats)')
            ax4.set_ylabel('Log Articulation Ratio')
            ax4.set_title('Articulation')
            ax4.grid(True)
            ax4.legend()

        plt.tight_layout()
        plt.savefig(save_path)


    def standardize_targets(self, score_notes: np.ndarray, perf_params: np.ndarray):
        """filter out outliers, scale the time parameters (beat_period and timing) into with 120 bpm. 
            When inferencing, the user will provide initial tempo to scale the time parameters"""
        # Filter out outliers 3 standard deviations from mean
        for param in ['beat_period', 'timing']:
            if param in perf_params.dtype.names:
                values = perf_params[param]
                mean = np.mean(values)
                std = np.std(values)
                mask = np.abs(values - mean) <= 3 * std
                perf_params = perf_params[mask]
                score_notes = score_notes[mask]

        # Scale time parameters to 120 BPM (0.5 seconds per beat)
        if 'beat_period' in perf_params.dtype.names:
            # Get average tempo
            avg_tempo = 60 / np.mean(perf_params['beat_period'])
            # Scale factor to convert to 120 BPM
            scale = avg_tempo / 120
            
            # Scale beat period and timing
            perf_params['beat_period'] = perf_params['beat_period'] * scale
            perf_params['timing'] = perf_params['timing'] * scale
                
        return score_notes, perf_params, avg_tempo
    
    def _compute_voice_layers(self, voice_pitches: np.ndarray) -> np.ndarray:
        """Compute voice layer (0=lowest, 1=middle, 2=highest) for each note in a voice"""
        if len(voice_pitches) == 0:
            return np.array([])
        
        # Sort pitches and assign layers
        sorted_indices = np.argsort(voice_pitches)
        layers = np.zeros(len(voice_pitches), dtype=int)
        
        # Assign layers: 0=lowest third, 1=middle third, 2=highest third
        third = len(voice_pitches) // 3
        layers[sorted_indices[:third]] = 0
        layers[sorted_indices[third:2*third]] = 1
        layers[sorted_indices[2*third:]] = 2
        
        return layers
    
    def _extract_pitch_features(self, voice_notes: np.ndarray, idx: int, pitch: int, 
                               piece_pitch_range: tuple, voice_stats: dict) -> dict:
        """Extract comprehensive pitch-related features"""
        features = {}
        
        # Basic pitch features
        features['pitch_class'] = pitch % 12
        features['octave'] = pitch // 12
        
        # Interval transitions (enhanced)
        if idx < len(voice_notes) - 1:
            features['pitch_interval_next'] = voice_notes[idx+1]['pitch'] - pitch
        else:
            features['pitch_interval_next'] = 0
            
        if idx > 0:
            features['pitch_interval_prev'] = pitch - voice_notes[idx-1]['pitch']
        else:
            features['pitch_interval_prev'] = 0
        
        # Multi-step interval context
        if idx < len(voice_notes) - 2:
            features['pitch_interval_2next'] = voice_notes[idx+2]['pitch'] - pitch
        else:
            features['pitch_interval_2next'] = 0
            
        if idx > 1:
            features['pitch_interval_2prev'] = pitch - voice_notes[idx-2]['pitch']
        else:
            features['pitch_interval_2prev'] = 0
            
        if idx < len(voice_notes) - 3:
            features['pitch_interval_3next'] = voice_notes[idx+3]['pitch'] - pitch
        else:
            features['pitch_interval_3next'] = 0
            
        if idx > 2:
            features['pitch_interval_3prev'] = pitch - voice_notes[idx-3]['pitch']
        else:
            features['pitch_interval_3prev'] = 0
        
        # Interval direction patterns
        features['interval_direction'] = self._get_interval_direction(features['pitch_interval_next'])
        features['interval_direction_2step'] = self._get_interval_direction(features['pitch_interval_2next'])
        features['interval_direction_3step'] = self._get_interval_direction(features['pitch_interval_3next'])
        
        # Melodic contour features
        features['melodic_contour_3step'] = self._get_melodic_contour(voice_notes, idx, 3)
        features['melodic_contour_5step'] = self._get_melodic_contour(voice_notes, idx, 5)
        
        # Pitch range context
        voice_min, voice_max = voice_stats['pitch_range']
        piece_min, piece_max = piece_pitch_range
        
        if voice_max > voice_min:
            features['pitch_relative_to_voice_range'] = (pitch - voice_min) / (voice_max - voice_min)
        else:
            features['pitch_relative_to_voice_range'] = 0.5
            
        if piece_max > piece_min:
            features['pitch_relative_to_piece_range'] = (pitch - piece_min) / (piece_max - piece_min)
        else:
            features['pitch_relative_to_piece_range'] = 0.5
        
        return features
    
    def _extract_voice_features(self, score_notes: np.ndarray, voice_notes: np.ndarray, 
                               idx: int, pitch: int, onset_beat: float, voice_layer: int) -> dict:
        """Extract voice-related features including cross-voice context"""
        features = {}
        
        # Voice layer information
        features['voice_layer'] = voice_layer
        features['voice_layer_relative'] = voice_layer / 2.0  # Normalize to 0-1
        
        # Find notes sounding at the same time (within a small tolerance)
        tolerance = 0.01  # 10ms tolerance
        sounding_notes = score_notes[
            (score_notes['onset_beat'] <= onset_beat) & 
            (score_notes['onset_beat'] + score_notes['duration_beat'] > onset_beat)
        ]
        
        # Notes above/below context
        notes_above = sounding_notes[sounding_notes['pitch'] > pitch]
        notes_below = sounding_notes[sounding_notes['pitch'] < pitch]
        
        features['notes_above_count'] = len(notes_above)
        features['notes_below_count'] = len(notes_below)
        
        if len(notes_above) > 0:
            features['notes_above_avg_pitch'] = notes_above['pitch'].mean()
            features['notes_above_max_pitch'] = notes_above['pitch'].max()
        else:
            features['notes_above_avg_pitch'] = pitch
            features['notes_above_max_pitch'] = pitch
            
        if len(notes_below) > 0:
            features['notes_below_avg_pitch'] = notes_below['pitch'].mean()
            features['notes_below_min_pitch'] = notes_below['pitch'].min()
        else:
            features['notes_below_avg_pitch'] = pitch
            features['notes_below_min_pitch'] = pitch
        
        # Voice density
        features['voice_density_at_onset'] = len(sounding_notes)
        features['voice_density_ratio'] = len(sounding_notes) / max(1, len(score_notes))
        
        # Cross-voice interval context
        if len(sounding_notes) > 0:
            highest_pitch = sounding_notes['pitch'].max()
            lowest_pitch = sounding_notes['pitch'].min()
            avg_pitch = sounding_notes['pitch'].mean()
            
            features['interval_to_highest_voice'] = highest_pitch - pitch
            features['interval_to_lowest_voice'] = lowest_pitch - pitch
            features['interval_to_voice_center'] = avg_pitch - pitch
        else:
            features['interval_to_highest_voice'] = 0
            features['interval_to_lowest_voice'] = 0
            features['interval_to_voice_center'] = 0
        
        return features
    
    def _extract_rhythmic_features(self, voice_notes: np.ndarray, idx: int, duration_beat: float,
                                  piece_avg_duration: float, voice_stats: dict, 
                                  piece_duration_ranks: np.ndarray) -> dict:
        """Extract comprehensive rhythmic features"""
        features = {}
        
        # Basic rhythmic features
        if idx < len(voice_notes) - 1:
            features['duration_ratio_next'] = voice_notes[idx+1]['duration_beat'] / duration_beat if duration_beat > 0 else 1.0
        else:
            features['duration_ratio_next'] = 1.0
            
        if idx > 0:
            features['duration_ratio_prev'] = duration_beat / voice_notes[idx-1]['duration_beat'] if voice_notes[idx-1]['duration_beat'] > 0 else 1.0
        else:
            features['duration_ratio_prev'] = 1.0
        
        # Enhanced rhythmic context
        features['rhythmic_context'] = self.compute_rhythmic_context(voice_notes['duration_beat'], idx)
        features['rhythmic_pattern_3step'] = self._get_rhythmic_pattern(voice_notes, idx, 3)
        features['rhythmic_pattern_5step'] = self._get_rhythmic_pattern(voice_notes, idx, 5)
        
        # Beat position features (simplified - would need measure information for full accuracy)
        onset_beat = voice_notes[idx]['onset_beat']
        features['beat_position_in_measure'] = (onset_beat % 4) / 4  # Assuming 4/4 time
        features['is_on_beat'] = abs(onset_beat % 1) < 0.1  # Within 0.1 beats of a whole beat
        features['is_on_downbeat'] = abs(onset_beat % 4) < 0.1  # Within 0.1 beats of measure start
        
        # Duration context
        features['duration_relative_to_voice_avg'] = duration_beat / voice_stats['avg_duration']
        features['duration_relative_to_piece_avg'] = duration_beat / piece_avg_duration
        features['duration_rank_in_voice'] = voice_stats['duration_ranks'][idx]
        
        # Find the corresponding piece duration rank (approximate)
        piece_idx = np.searchsorted(np.sort(voice_notes['duration_beat']), duration_beat)
        if piece_idx < len(piece_duration_ranks):
            features['duration_rank_in_piece'] = piece_duration_ranks[piece_idx]
        else:
            features['duration_rank_in_piece'] = 0.5
        

        
        return features
    
    def _extract_phrase_features(self, voice_notes: np.ndarray, idx: int, 
                                phrase_boundaries: List[int]) -> dict:
        """Extract phrase-related features"""
        features = {}
        
        # Find which phrase this note belongs to
        phrase_idx = 0
        for j, boundary in enumerate(phrase_boundaries[:-1]):
            if boundary <= idx < phrase_boundaries[j+1]:
                phrase_idx = j
                break
        
        phrase_start = phrase_boundaries[phrase_idx]
        phrase_end = phrase_boundaries[phrase_idx + 1]
        
        # Basic phrase features
        features['position_in_phrase'] = (idx - phrase_start) / max(1, phrase_end - phrase_start)
        features['phrase_length'] = phrase_end - phrase_start
        

        
        # Implication-Realization analysis
        ir_label, ir_closure = self.compute_ir_analysis(voice_notes['pitch'], idx)
        features['ir_label'] = ir_label
        features['ir_closure'] = ir_closure
        
        return features
    
    def _get_interval_direction(self, interval: int) -> str:
        """Get direction of interval: 'up', 'down', or 'same'"""
        if interval > 0:
            return "up"
        elif interval < 0:
            return "down"
        else:
            return "same"
    
    def _get_melodic_contour(self, voice_notes: np.ndarray, idx: int, steps: int) -> str:
        """Get melodic contour pattern for given number of steps"""
        if idx + steps >= len(voice_notes):
            return "incomplete"
        
        contour = []
        for i in range(steps):
            if idx + i + 1 < len(voice_notes):
                interval = voice_notes[idx + i + 1]['pitch'] - voice_notes[idx + i]['pitch']
                contour.append(self._get_interval_direction(interval))
        
        return "-".join(contour)
    
    def _get_rhythmic_pattern(self, voice_notes: np.ndarray, idx: int, steps: int) -> str:
        """Get rhythmic pattern for given number of steps"""
        if idx + steps >= len(voice_notes):
            return "incomplete"
        
        pattern = []
        for i in range(steps):
            if idx + i < len(voice_notes):
                duration = voice_notes[idx + i]['duration_beat']
                if duration < 0.5:
                    pattern.append("s")  # short
                elif duration < 1.0:
                    pattern.append("m")  # medium
                else:
                    pattern.append("l")  # long
        
        return "-".join(pattern)
    

        

class MidiHumFeatureEngineer:
    """
    Feature engineering inspired by the midihum/midi_to_df_conversion.py _add_engineered_features function.
    This class augments a note array (structured numpy array) of note features with additional engineered features for expressive performance modeling.
    """
    def __init__(self):
        pass

    def _note_array_to_midihum_df(self, note_array):
        """
        Convert a structured numpy note array to a DataFrame with columns compatible with midihum feature engineering.
        Computes chord_character_pressed and chord_size_pressed for each note using chord_identifier.chord_attributes.
        """
        import numpy as np
        import pandas as pd
        # Map available fields
        df = pd.DataFrame({
            'time': note_array['onset_beat'],
            'sustain': note_array['duration_beat'],
            'pitch': note_array['pitch'],
            'midi_track_index': note_array['voice'] if 'voice' in note_array.dtype.fields else 0,
            'name': note_array['id'] if 'id' in note_array.dtype.fields else None,
        })
        # Fill missing fields with default values
        df['velocity'] = 64  # Default MIDI velocity if not available
        df['pitch_class'] = df['pitch'] % 12
        df['octave'] = df['pitch'] // 12
        df['avg_pitch_pressed'] = df['pitch']  # Placeholder: single note
        df['nearness_to_end'] = 0.0  # Placeholder
        df['nearness_to_midpoint'] = 0.0  # Placeholder
        df['interval_from_pressed'] = 0.0  # Placeholder
        df['interval_from_released'] = 0.0  # Placeholder
        df['num_played_notes_pressed'] = 1  # Placeholder
        df['follows_pause'] = 0  # Placeholder

        # Compute chord_character_pressed and chord_size_pressed for each note
        chord_char_list = []
        chord_size_list = []
        # Precompute note offsets
        offsets = df['time'] + df['sustain']
        for idx, row in df.iterrows():
            onset = row['time']
            # Find all notes sounding at this onset
            sounding = df[(df['time'] <= onset) & (offsets > onset)]
            curr_pitches = sounding['pitch'].values.astype(int)
            attrs = chord_attributes(curr_pitches)
            if attrs is not None:
                chord_char_list.append(attrs[0] if attrs[0] is not None else 'none')
                chord_size_list.append(attrs[1] if attrs[1] is not None else 'none')
            else:
                chord_char_list.append('none')
                chord_size_list.append('none')
        df['chord_character_pressed'] = chord_char_list
        df['chord_size_pressed'] = chord_size_list
        return df

    def add_midihum_features(self, note_array, with_extra_features: bool = False):
        """
        Add engineered features to a structured numpy note array.
        Returns a DataFrame with additional features.
        """
        import numpy as np
        import pandas as pd
        from sklearn import preprocessing

        df = self._note_array_to_midihum_df(note_array)
        new_cols = {}

        # calculate "true" chord character and size by bunching all samples within 5 time units together
        df["chord_character"] = df.groupby(
            np.floor(df.time / 5) * 5
        ).chord_character_pressed.transform("last")
        df["chord_size"] = df.groupby(
            np.floor(df.time / 5) * 5
        ).chord_size_pressed.transform("last")

        # get time elapsed since last note event(s)
        df["time_since_last_pressed"] = (df.time - df.time.shift()).fillna(0)
        df["time_since_last_released"] = (
            df.time - (df.time.shift() + df.sustain.shift())
        ).fillna(0)

        # get time elapsed since various further events
        for cat in [
            "pitch_class",
            "octave",
            "follows_pause",
            "chord_character",
            "chord_size",
        ]:
            col_name = f"time_since_{cat}"
            col = pd.Series(
                preprocessing.scale(
                    (df.time - df.groupby(cat)["time"].shift()).fillna(0).values
                )
            )
            new_cols[col_name] = col
            new_cols[f"log_{col_name}"] = pd.Series(np.log(col + 1))

        # add some abs cols
        for col in ["interval_from_pressed", "interval_from_released"]:
            base = new_cols[col] if col in new_cols else df[col]
            new_cols[f"abs_{col}"] = np.abs(base)

        # add some log cols
        for col in [
            "time_since_chord_character",
            "time_since_chord_size",
            "time_since_follows_pause",
            "time_since_octave",
            "time_since_pitch_class",
        ]:
            base = new_cols[col] if col in new_cols else df[col]
            new_cols[f"log_{col}"] = pd.Series(np.log10(np.abs(base) + 1))
        for col in [
            "sustain",
            "time_since_last_pressed",
            "time_since_last_released",
            "abs_interval_from_pressed",
            "abs_interval_from_released",
        ]:
            base = new_cols[col] if col in new_cols else df[col]
            new_cols[f"log_{col}"] = pd.Series(np.log(np.abs(base) + 1))

        # calculate some simple moving averages
        sma_aggs = {
            "pitch": ["mean", "min", "max", "std"],
            "log_sustain": ["mean", "min", "max", "std"],
            "interval_from_pressed": ["mean", "min", "max", "std"],
            "log_time_since_last_pressed": ["mean", "min", "max", "std"],
            "log_time_since_follows_pause": ["mean", "min", "max", "std"],
        }
        sma_windows = [15, 30, 75]
        for col, funcs in sma_aggs.items():
            base = new_cols[col] if col in new_cols else df[col]
            for window in sma_windows:
                for func in funcs:
                    sma = base.rolling(window).agg(func).bfill()
                    new_cols[f"{col}_sma_{func}_{window}"] = sma
                    fwd_sma = base[::-1].rolling(window).agg(func).bfill()[::-1]
                    new_cols[f"{col}_fwd_sma_{func}_{window}"] = fwd_sma

                    if col != "follows_pause":
                        new_cols[f"{col}_sma_{func}_{window}_oscillator"] = base - sma
                        new_cols[f"{col}_fwd_sma_{func}_{window}_oscillator"] = (
                            base - fwd_sma
                        )

        # add ichimoku indicators
        for col in [
            "pitch",
            "log_sustain",
            "interval_from_released",
            "interval_from_pressed",
        ]:
            base = new_cols[col] if col in new_cols else df[col]
            tenkan_sen = (base.rolling(9).max() + base.rolling(9).min()).bfill() / 2.0
            kijun_sen = (base.rolling(26).max() + base.rolling(26).min()).bfill() / 2.0
            senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
            senkou_span_b = (base.rolling(52).max() + base.rolling(52).min()).bfill() / 2.0

            new_cols[f"{col}_tenkan_sen"] = tenkan_sen
            new_cols[f"{col}_kijun_sen"] = kijun_sen
            new_cols[f"{col}_senkou_span_a"] = senkou_span_a
            new_cols[f"{col}_senkou_span_b"] = senkou_span_b
            new_cols[f"{col}_chikou_span"] = base.shift(26).bfill()
            new_cols[f"{col}_cloud_is_green"] = senkou_span_a - senkou_span_b

            new_cols[f"{col}_relative_to_tenkan_sen"] = base - tenkan_sen
            new_cols[f"{col}_relative_to_kijun_sen"] = base - kijun_sen
            new_cols[f"{col}_tenkan_sen_relative_to_kijun_sen"] = tenkan_sen - kijun_sen
            new_cols[f"{col}_relative_to_chikou_span"] = base - base.shift(26).bfill()
            new_cols[f"{col}_relative_to_cloud"] = (
                base - (senkou_span_a + senkou_span_b) / 2.0
            )

        # (Optional: add percent change, lag, and aggregate features as in the original if needed)

        for name, new_col in new_cols.items():
            if not pd.api.types.is_numeric_dtype(new_col):
                continue
            assert not np.any(np.isnan(new_col)), (name, new_col)
            assert np.all(np.isfinite(new_col)), (name, new_col)

        return pd.concat(
            [df] + [col.rename(name) for name, col in new_cols.items()], axis=1
        )


