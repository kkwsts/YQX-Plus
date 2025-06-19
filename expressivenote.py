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
