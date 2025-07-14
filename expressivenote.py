from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ExpressiveNote:
    """Container for a note with its context features and expressive targets"""
    
    #########################################################
    ##### BASIC SCORE FEATURES ##############################
    #########################################################
    pitch: int
    onset_beat: float
    duration_beat: float
    voice: int
    
    #########################################################
    ##### PITCH FEATURES ####################################
    #########################################################
    # Basic pitch features
    pitch_class: Optional[int] = None  # 0-11 pitch class
    octave: Optional[int] = None  # Octave number
    
    # Interval transitions (enhanced)
    pitch_interval_next: Optional[int] = None  # semitones to next note (original)
    pitch_interval_prev: Optional[int] = None  # semitones from previous note
    
    # Multi-step interval context
    pitch_interval_2next: Optional[int] = None  # semitones to note after next
    pitch_interval_2prev: Optional[int] = None  # semitones from note before previous
    pitch_interval_3next: Optional[int] = None  # semitones to note 3 steps ahead
    pitch_interval_3prev: Optional[int] = None  # semitones from note 3 steps back
    
    # Interval direction patterns
    interval_direction: Optional[str] = None  # "up", "down", "same"
    interval_direction_2step: Optional[str] = None  # 2-step pattern
    interval_direction_3step: Optional[str] = None  # 3-step pattern
    
    # Melodic contour features
    melodic_contour_3step: Optional[str] = None  # e.g., "up-up", "up-down", "down-up"
    melodic_contour_5step: Optional[str] = None  # 5-step contour pattern
    
    # Pitch range context
    pitch_relative_to_voice_range: Optional[float] = None  # 0-1, position in voice's pitch range
    pitch_relative_to_piece_range: Optional[float] = None  # 0-1, position in piece's pitch range
    
    #########################################################
    ##### VOICE FEATURES ####################################
    #########################################################
    # Voice layer information
    voice_layer: Optional[int] = None  # Which layer (0=lowest, 1=middle, 2=highest)
    voice_layer_relative: Optional[float] = None  # 0-1, relative position in voice layers
    
    # Notes above/below context
    notes_above_count: Optional[int] = None  # Number of notes above this note at onset
    notes_below_count: Optional[int] = None  # Number of notes below this note at onset
    notes_above_avg_pitch: Optional[float] = None  # Average pitch of notes above
    notes_below_avg_pitch: Optional[float] = None  # Average pitch of notes below
    notes_above_max_pitch: Optional[float] = None  # Highest pitch above
    notes_below_min_pitch: Optional[float] = None  # Lowest pitch below
    
    # Voice density
    voice_density_at_onset: Optional[float] = None  # Total notes sounding at onset
    voice_density_ratio: Optional[float] = None  # This voice's notes / total notes at onset
    
    # Cross-voice interval context
    interval_to_highest_voice: Optional[int] = None  # Interval to highest sounding note
    interval_to_lowest_voice: Optional[int] = None  # Interval to lowest sounding note
    interval_to_voice_center: Optional[int] = None  # Interval to average pitch of all voices
    
    #########################################################
    ##### RHYTHMIC FEATURES #################################
    #########################################################
    # Basic rhythmic features
    duration_ratio_next: Optional[float] = None  # duration ratio to next note (original)
    duration_ratio_prev: Optional[float] = None  # duration ratio from previous note
    
    # Enhanced rhythmic context
    rhythmic_context: Optional[str] = None  # e.g., "s-s-l" (short-short-long)
    rhythmic_pattern_3step: Optional[str] = None  # 3-step rhythmic pattern
    rhythmic_pattern_5step: Optional[str] = None  # 5-step rhythmic pattern
    
    # Beat position features
    beat_position_in_measure: Optional[float] = None  # 0-1, position within measure
    beat_position_in_phrase: Optional[float] = None  # 0-1, position within phrase
    is_on_beat: Optional[bool] = None  # Whether note starts on a strong beat
    is_on_downbeat: Optional[bool] = None  # Whether note starts on measure downbeat
    
    # Duration context
    duration_relative_to_voice_avg: Optional[float] = None  # Duration / voice average duration
    duration_relative_to_piece_avg: Optional[float] = None  # Duration / piece average duration
    duration_rank_in_voice: Optional[float] = None  # 0-1, rank of duration within voice
    duration_rank_in_piece: Optional[float] = None  # 0-1, rank of duration within piece
    
    #########################################################
    ##### PHRASE FEATURES ###################################
    #########################################################
    # Basic phrase features
    position_in_phrase: Optional[float] = None  # 0.0 to 1.0 (original)
    phrase_length: Optional[int] = None  # Number of notes in phrase
    
    # Implication-Realization analysis
    ir_label: Optional[str] = None  # Implication-Realization label
    ir_closure: Optional[float] = None  # Musical closure measure
    
    #########################################################
    ##### EXPRESSIVE TARGETS ################################
    #########################################################
    # Expressive targets (None for inference)
    beat_period: Optional[float] = None  # Beat period 
    timing: Optional[float] = None  # Timing 
    velocity: Optional[float] = None  # Loudness (0.0 to 1.0)
    articulation_log: Optional[float] = None  # Log-Articulation ratio 

    #########################################################
    ##### MIDIHUM FEATURES ##################################
    #########################################################
    
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
