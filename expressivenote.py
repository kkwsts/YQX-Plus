from dataclasses import dataclass
from typing import Optional

@dataclass
class ExpressiveNote:
    """Container for a note with its context features and expressive targets"""
    # Score features
    pitch: int
    onset_beat: float
    duration_beat: float
    voice: int
    
    # Context features
    pitch_interval: int  # semitones to next note
    duration_ratio: float  # duration ratio to next note
    rhythmic_context: str  # e.g., "s-s-l" (short-short-long)
    ir_label: str  # Implication-Realization label
    ir_closure: float  # Musical closure measure
    position_in_phrase: float  # 0.0 to 1.0
    
    # Expressive targets (None for inference)
    beat_period: Optional[float] = None  # Beat period 
    timing: Optional[float] = None  # Timing 
    velocity: Optional[float] = None  # Loudness (0.0 to 1.0)
    articulation_log: Optional[float] = None  # Log-Articulation ratio 

    #########################################################
    ##### MIDIHUM FEATURES ##################################
    #########################################################
    
    pitch_class: Optional[int] = None  # 0-11 pitch class
    octave: Optional[int] = None  # Octave number
    follows_pause: Optional[bool] = None  # Whether note follows a pause
    
    # Chord context features
    chord_character_pressed: Optional[str] = None  # e.g., 'major', 'minor', etc.
    chord_size_pressed: Optional[str] = None  # e.g., 'triad', 'seventh', etc.
    chord_character: Optional[str] = None  # Grouped by 5 time units
    chord_size: Optional[str] = None  # Grouped by 5 time units
    num_played_notes_pressed: Optional[int] = None  # Number of simultaneous notes
    avg_pitch_pressed: Optional[float] = None  # Average pitch of simultaneous notes
    
    # Timing context features
    time_since_last_pressed: Optional[float] = None  # Time since last note onset
    time_since_last_released: Optional[float] = None  # Time since last note release
    
    # Time since various events (normalized)
    time_since_pitch_class: Optional[float] = None  # Time since last same pitch class
    time_since_octave: Optional[float] = None  # Time since last same octave
    time_since_follows_pause: Optional[float] = None  # Time since last pause
    time_since_chord_character: Optional[float] = None  # Time since last same chord type
    time_since_chord_size: Optional[float] = None  # Time since last same chord size
    
    # Log versions of time features
    log_time_since_pitch_class: Optional[float] = None
    log_time_since_octave: Optional[float] = None
    log_time_since_follows_pause: Optional[float] = None
    log_time_since_chord_character: Optional[float] = None
    log_time_since_chord_size: Optional[float] = None
    log_time_since_last_pressed: Optional[float] = None
    log_time_since_last_released: Optional[float] = None
    
    # Interval features
    interval_from_pressed: Optional[float] = None  # Interval to simultaneous notes
    interval_from_released: Optional[float] = None  # Interval to previous notes
    abs_interval_from_pressed: Optional[float] = None
    abs_interval_from_released: Optional[float] = None
    log_abs_interval_from_pressed: Optional[float] = None
    log_abs_interval_from_released: Optional[float] = None
    
    # Moving average features (15, 30, 75 windows)
    # Pitch SMA
    pitch_sma_mean_15: Optional[float] = None
    pitch_sma_min_15: Optional[float] = None
    pitch_sma_max_15: Optional[float] = None
    pitch_sma_std_15: Optional[float] = None
    pitch_sma_mean_30: Optional[float] = None
    pitch_sma_min_30: Optional[float] = None
    pitch_sma_max_30: Optional[float] = None
    pitch_sma_std_30: Optional[float] = None
    pitch_sma_mean_75: Optional[float] = None
    pitch_sma_min_75: Optional[float] = None
    pitch_sma_max_75: Optional[float] = None
    pitch_sma_std_75: Optional[float] = None
    
    # Forward Pitch SMA
    pitch_fwd_sma_mean_15: Optional[float] = None
    pitch_fwd_sma_min_15: Optional[float] = None
    pitch_fwd_sma_max_15: Optional[float] = None
    pitch_fwd_sma_std_15: Optional[float] = None
    pitch_fwd_sma_mean_30: Optional[float] = None
    pitch_fwd_sma_min_30: Optional[float] = None
    pitch_fwd_sma_max_30: Optional[float] = None
    pitch_fwd_sma_std_30: Optional[float] = None
    pitch_fwd_sma_mean_75: Optional[float] = None
    pitch_fwd_sma_min_75: Optional[float] = None
    pitch_fwd_sma_max_75: Optional[float] = None
    pitch_fwd_sma_std_75: Optional[float] = None
    
    # Pitch SMA Oscillators
    pitch_sma_mean_15_oscillator: Optional[float] = None
    pitch_sma_min_15_oscillator: Optional[float] = None
    pitch_sma_max_15_oscillator: Optional[float] = None
    pitch_sma_std_15_oscillator: Optional[float] = None
    pitch_sma_mean_30_oscillator: Optional[float] = None
    pitch_sma_min_30_oscillator: Optional[float] = None
    pitch_sma_max_30_oscillator: Optional[float] = None
    pitch_sma_std_30_oscillator: Optional[float] = None
    pitch_sma_mean_75_oscillator: Optional[float] = None
    pitch_sma_min_75_oscillator: Optional[float] = None
    pitch_sma_max_75_oscillator: Optional[float] = None
    pitch_sma_std_75_oscillator: Optional[float] = None
    
    # Forward Pitch SMA Oscillators
    pitch_fwd_sma_mean_15_oscillator: Optional[float] = None
    pitch_fwd_sma_min_15_oscillator: Optional[float] = None
    pitch_fwd_sma_max_15_oscillator: Optional[float] = None
    pitch_fwd_sma_std_15_oscillator: Optional[float] = None
    pitch_fwd_sma_mean_30_oscillator: Optional[float] = None
    pitch_fwd_sma_min_30_oscillator: Optional[float] = None
    pitch_fwd_sma_max_30_oscillator: Optional[float] = None
    pitch_fwd_sma_std_30_oscillator: Optional[float] = None
    pitch_fwd_sma_mean_75_oscillator: Optional[float] = None
    pitch_fwd_sma_min_75_oscillator: Optional[float] = None
    pitch_fwd_sma_max_75_oscillator: Optional[float] = None
    pitch_fwd_sma_std_75_oscillator: Optional[float] = None
    
    # Log Sustain SMA features (similar pattern for all windows)
    log_sustain_sma_mean_15: Optional[float] = None
    log_sustain_sma_min_15: Optional[float] = None
    log_sustain_sma_max_15: Optional[float] = None
    log_sustain_sma_std_15: Optional[float] = None
    log_sustain_sma_mean_30: Optional[float] = None
    log_sustain_sma_min_30: Optional[float] = None
    log_sustain_sma_max_30: Optional[float] = None
    log_sustain_sma_std_30: Optional[float] = None
    log_sustain_sma_mean_75: Optional[float] = None
    log_sustain_sma_min_75: Optional[float] = None
    log_sustain_sma_max_75: Optional[float] = None
    log_sustain_sma_std_75: Optional[float] = None
    
    # Forward Log Sustain SMA
    log_sustain_fwd_sma_mean_15: Optional[float] = None
    log_sustain_fwd_sma_min_15: Optional[float] = None
    log_sustain_fwd_sma_max_15: Optional[float] = None
    log_sustain_fwd_sma_std_15: Optional[float] = None
    log_sustain_fwd_sma_mean_30: Optional[float] = None
    log_sustain_fwd_sma_min_30: Optional[float] = None
    log_sustain_fwd_sma_max_30: Optional[float] = None
    log_sustain_fwd_sma_std_30: Optional[float] = None
    log_sustain_fwd_sma_mean_75: Optional[float] = None
    log_sustain_fwd_sma_min_75: Optional[float] = None
    log_sustain_fwd_sma_max_75: Optional[float] = None
    log_sustain_fwd_sma_std_75: Optional[float] = None
    
    # Log Sustain SMA Oscillators
    log_sustain_sma_mean_15_oscillator: Optional[float] = None
    log_sustain_sma_min_15_oscillator: Optional[float] = None
    log_sustain_sma_max_15_oscillator: Optional[float] = None
    log_sustain_sma_std_15_oscillator: Optional[float] = None
    log_sustain_sma_mean_30_oscillator: Optional[float] = None
    log_sustain_sma_min_30_oscillator: Optional[float] = None
    log_sustain_sma_max_30_oscillator: Optional[float] = None
    log_sustain_sma_std_30_oscillator: Optional[float] = None
    log_sustain_sma_mean_75_oscillator: Optional[float] = None
    log_sustain_sma_min_75_oscillator: Optional[float] = None
    log_sustain_sma_max_75_oscillator: Optional[float] = None
    log_sustain_sma_std_75_oscillator: Optional[float] = None
    
    # Forward Log Sustain SMA Oscillators
    log_sustain_fwd_sma_mean_15_oscillator: Optional[float] = None
    log_sustain_fwd_sma_min_15_oscillator: Optional[float] = None
    log_sustain_fwd_sma_max_15_oscillator: Optional[float] = None
    log_sustain_fwd_sma_std_15_oscillator: Optional[float] = None
    log_sustain_fwd_sma_mean_30_oscillator: Optional[float] = None
    log_sustain_fwd_sma_min_30_oscillator: Optional[float] = None
    log_sustain_fwd_sma_max_30_oscillator: Optional[float] = None
    log_sustain_fwd_sma_std_30_oscillator: Optional[float] = None
    log_sustain_fwd_sma_mean_75_oscillator: Optional[float] = None
    log_sustain_fwd_sma_min_75_oscillator: Optional[float] = None
    log_sustain_fwd_sma_max_75_oscillator: Optional[float] = None
    log_sustain_fwd_sma_std_75_oscillator: Optional[float] = None
    
    # Technical indicators for pitch
    pitch_tenkan_sen: Optional[float] = None  # 9-period moving average
    pitch_kijun_sen: Optional[float] = None  # 26-period moving average
    pitch_senkou_span_a: Optional[float] = None  # (tenkan + kijun) / 2
    pitch_senkou_span_b: Optional[float] = None  # 52-period moving average
    pitch_chikou_span: Optional[float] = None  # Current price 26 periods ahead
    pitch_cloud_is_green: Optional[float] = None  # senkou_a > senkou_b
    pitch_relative_to_tenkan_sen: Optional[float] = None
    pitch_relative_to_kijun_sen: Optional[float] = None
    pitch_tenkan_sen_relative_to_kijun_sen: Optional[float] = None
    pitch_relative_to_chikou_span: Optional[float] = None
    pitch_relative_to_cloud: Optional[float] = None
    
    # Technical indicators for log_sustain
    log_sustain_tenkan_sen: Optional[float] = None
    log_sustain_kijun_sen: Optional[float] = None
    log_sustain_senkou_span_a: Optional[float] = None
    log_sustain_senkou_span_b: Optional[float] = None
    log_sustain_chikou_span: Optional[float] = None
    log_sustain_cloud_is_green: Optional[float] = None
    log_sustain_relative_to_tenkan_sen: Optional[float] = None
    log_sustain_relative_to_kijun_sen: Optional[float] = None
    log_sustain_tenkan_sen_relative_to_kijun_sen: Optional[float] = None
    log_sustain_relative_to_chikou_span: Optional[float] = None
    log_sustain_relative_to_cloud: Optional[float] = None
    
    # Technical indicators for interval_from_released
    interval_from_released_tenkan_sen: Optional[float] = None
    interval_from_released_kijun_sen: Optional[float] = None
    interval_from_released_senkou_span_a: Optional[float] = None
    interval_from_released_senkou_span_b: Optional[float] = None
    interval_from_released_chikou_span: Optional[float] = None
    interval_from_released_cloud_is_green: Optional[float] = None
    interval_from_released_relative_to_tenkan_sen: Optional[float] = None
    interval_from_released_relative_to_kijun_sen: Optional[float] = None
    interval_from_released_tenkan_sen_relative_to_kijun_sen: Optional[float] = None
    interval_from_released_relative_to_chikou_span: Optional[float] = None
    interval_from_released_relative_to_cloud: Optional[float] = None
    
    # Technical indicators for interval_from_pressed
    interval_from_pressed_tenkan_sen: Optional[float] = None
    interval_from_pressed_kijun_sen: Optional[float] = None
    interval_from_pressed_senkou_span_a: Optional[float] = None
    interval_from_pressed_senkou_span_b: Optional[float] = None
    interval_from_pressed_chikou_span: Optional[float] = None
    interval_from_pressed_cloud_is_green: Optional[float] = None
    interval_from_pressed_relative_to_tenkan_sen: Optional[float] = None
    interval_from_pressed_relative_to_kijun_sen: Optional[float] = None
    interval_from_pressed_tenkan_sen_relative_to_kijun_sen: Optional[float] = None
    interval_from_pressed_relative_to_chikou_span: Optional[float] = None
    interval_from_pressed_relative_to_cloud: Optional[float] = None

    def get_targets(self):
        return [self.beat_period, self.timing, self.velocity, self.articulation_log]
