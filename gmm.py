import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hook
import warnings

from expressivenote import ExpressiveNote

class BayesianExpressiveModel:
    """Bayesian model for predicting expressive parameters"""
    
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.models = {}  # One model per target variable
        self.trained = False
    
    def train(self, training_notes: List[List[ExpressiveNote]], feature_extractor):
        """Train the Bayesian model on training data"""
        print("Training YQX model...")
        
        # Flatten all notes
        all_notes = []
        for piece_notes in training_notes:
            all_notes.extend(piece_notes)
        
        # Filter out notes without targets
        training_notes_filtered = [note for note in all_notes if 
                                 note.beat_period is not None and
                                 note.timing is not None and
                                 note.velocity is not None and
                                 note.articulation_log is not None]
        
        print(f"Training on {len(training_notes_filtered)} notes")
        
        # Extract and encode features
        X = feature_extractor.encode_features(training_notes_filtered, fit=True)

        # Extract targets
        targets = {
            'beat_period': np.array([note.beat_period for note in training_notes_filtered]),
            'timing': np.array([note.timing for note in training_notes_filtered]),
            'velocity': np.array([note.velocity for note in training_notes_filtered]),
            'articulation_log': np.array([note.articulation_log for note in training_notes_filtered])
        }
        
        # Train separate models for each target
        for target_name, y in targets.items():
            print(f"Training {target_name} model...")
            
            # Train Gaussian Mixture Model
            model = GaussianMixture(n_components=self.n_components, random_state=42)
            
            # Combine features and targets for joint modeling
            joint_data = np.column_stack([X, y])
            model.fit(joint_data)
            
            self.models[target_name] = model
        
        self.trained = True
        print("Training completed!")
    
    def predict(self, notes: List[ExpressiveNote], feature_extractor) -> List[ExpressiveNote]:
        """Predict expressive parameters for new notes"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract and encode features
        X = feature_extractor.encode_features(notes, fit=False)
        
        # Predict each target
        predicted_parameters = []
        for i, note in enumerate(notes):
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
                position_in_phrase=note.position_in_phrase
            )
            
            # Predict each target variable
            for target_name in ['beat_period', 'timing', 'velocity', 'articulation_log']:
                model = self.models[target_name]
                
                # Use conditional expectation given features
                # Approximate by finding most likely component and using its mean
                features = X[i:i+1]
                
                # Find most likely component
                log_probs = []
                for comp in range(self.n_components):
                    mean = model.means_[comp]
                    cov = model.covariances_[comp]
                    
                    # Compute log probability of features under this component
                    feature_mean = mean[:-1]  # All but last dimension (target)
                    feature_cov = cov[:-1, :-1]
                    
                    try:
                        inv_cov = np.linalg.inv(feature_cov)
                        diff = features[0] - feature_mean
                        log_prob = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
                        log_probs.append(log_prob + np.log(model.weights_[comp]))
                    except:
                        log_probs.append(-np.inf)
                
                best_comp = np.argmax(log_probs)
                
                # Predict target using conditional mean
                target_mean = model.means_[best_comp, -1]  # Last dimension is target
                target_pred = target_mean
                
                # Apply reasonable ranges and assign to note
                if target_name == 'beat_period':
                    # target_pred = np.clip(target_pred, 0.3, 3.0)
                    new_note.beat_period = target_pred
                elif target_name == 'timing':
                    # target_pred = np.clip(target_pred, -0.5, 0.5)
                    new_note.timing = target_pred
                elif target_name == 'velocity':
                    # target_pred = int(np.clip(target_pred, 0, 1))
                    new_note.velocity = target_pred
                elif target_name == 'articulation_log':
                    # target_pred = np.clip(target_pred, -2.0, 1.0)
                    new_note.articulation_log = target_pred
            
            predicted_parameters.append(new_note)
        
        return predicted_parameters
    
    def save(self, filepath: str):
        """Save trained model"""
        model_data = {
            'models': self.models,
            'n_components': self.n_components,
            'trained': self.trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.n_components = model_data['n_components']
        self.trained = model_data['trained']





