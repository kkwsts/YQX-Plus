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
        self.scalers = {}
        self.feature_encoders = {}
        self.trained = False
    
    def _encode_categorical_features(self, notes: List[ExpressiveNote], fit: bool = False) -> np.ndarray:
        """Encode categorical features to numerical"""
        categorical_features = []
        continuous_features = []
        
        for note in notes:
            categorical_features.append([note.rhythmic_context, note.ir_label])
            continuous_features.append([
                note.pitch,
                note.duration_beat,
                note.pitch_interval,
                note.duration_ratio,
                note.ir_closure,
                note.position_in_phrase
            ])
        
        # Encode categorical features
        if fit:
            # Create encodings
            unique_rhythmic = list(set(f[0] for f in categorical_features))
            unique_ir = list(set(f[1] for f in categorical_features))
            
            self.feature_encoders['rhythmic_context'] = {v: i for i, v in enumerate(unique_rhythmic)}
            self.feature_encoders['ir_label'] = {v: i for i, v in enumerate(unique_ir)}
        
        # Apply encodings
        encoded_categorical = []
        for features in categorical_features:
            rhythmic_encoded = self.feature_encoders['rhythmic_context'].get(features[0], 0)
            ir_encoded = self.feature_encoders['ir_label'].get(features[1], 0)
            encoded_categorical.append([rhythmic_encoded, ir_encoded])
        
        # Combine features
        all_features = np.hstack([np.array(encoded_categorical), np.array(continuous_features)])
        
        return all_features
    
    def train(self, training_notes: List[List[ExpressiveNote]]):
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
        
        # Extract features
        X = self._encode_categorical_features(training_notes_filtered, fit=True)
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)

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
            
            # Scale targets
            self.scalers[target_name] = StandardScaler()
            y_scaled = self.scalers[target_name].fit_transform(y.reshape(-1, 1)).flatten()
            
            # Train Gaussian Mixture Model
            model = GaussianMixture(n_components=self.n_components, random_state=42)
            
            # Combine features and targets for joint modeling
            joint_data = np.column_stack([X_scaled, y_scaled])
            model.fit(joint_data)
            
            self.models[target_name] = model
        
        self.trained = True
        print("Training completed!")
    
    def predict(self, notes: List[ExpressiveNote]) -> List[ExpressiveNote]:
        """Predict expressive parameters for new notes"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        X = self._encode_categorical_features(notes, fit=False)
        X_scaled = self.scalers['features'].transform(X)
        
        # Predict each target
        predicted_parameters = []
        for i, note in enumerate(notes):
            new_note = ExpressiveNote(
                pitch=note.pitch,
                onset_beat=note.onset_beat,
                duration_beat=note.duration_beat,
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
                features = X_scaled[i:i+1]
                
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
                target_pred_scaled = target_mean
                
                # Unscale prediction
                target_pred = self.scalers[target_name].inverse_transform([[target_pred_scaled]])[0, 0]
                
                # Apply reasonable ranges and assign to note
                if target_name == 'beat_period':
                    target_pred = np.clip(target_pred, 0.3, 3.0)
                    new_note.beat_period = target_pred
                elif target_name == 'timing':
                    target_pred = np.clip(target_pred, -0.5, 0.5)
                    new_note.timing = target_pred
                elif target_name == 'velocity':
                    target_pred = int(np.clip(target_pred, 1, 127))
                    new_note.velocity = target_pred
                elif target_name == 'articulation_log':
                    target_pred = np.clip(target_pred, -2.0, 1.0)
                    new_note.articulation_log = target_pred
            
            predicted_parameters.append(new_note)
        
        return predicted_parameters
    
    def save(self, filepath: str):
        """Save trained model"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_encoders': self.feature_encoders,
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
        self.scalers = model_data['scalers']
        self.feature_encoders = model_data['feature_encoders']
        self.n_components = model_data['n_components']
        self.trained = model_data['trained']





