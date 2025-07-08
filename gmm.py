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
    
    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.models = {}  # One model per target variable
        self.trained = False
        self.feature_scaler = None  # Store scaler in model
    
    def train(self, context_features: np.ndarray, targets: np.ndarray, **kwargs):
        """Train the Bayesian model on pre-extracted features and targets"""
        print("Training GMM model...")
        print(f"Training on {len(context_features)} samples")
        
        # Split targets into separate arrays
        target_dict = {
            'beat_period': targets[:, 0],
            'timing': targets[:, 1], 
            'velocity': targets[:, 2],
            'articulation_log': targets[:, 3]
        }
        
        # Train separate models for each target
        for target_name, y in target_dict.items():
            print(f"Training {target_name} model...")
            
            # Train Gaussian Mixture Model
            model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
            
            # Combine features and targets for joint modeling
            joint_data = np.column_stack([context_features, y])
            model.fit(joint_data)
            
            self.models[target_name] = model
        
        self.trained = True
        print("GMM training completed!")
    
    def predict(self, context_features: np.ndarray) -> np.ndarray:
        """Predict targets for pre-extracted features"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Initialize predictions array
        predictions = np.zeros((len(context_features), 4))
        
        # Predict each target variable
        target_names = ['beat_period', 'timing', 'velocity', 'articulation_log']
        for target_idx, target_name in enumerate(target_names):
            model = self.models[target_name]
            
            for i in range(len(context_features)):
                # Use conditional expectation given features
                # Approximate by finding most likely component and using its mean
                features = context_features[i:i+1]
                
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
                predictions[i, target_idx] = target_mean
        
        return predictions
    
    def save(self, filepath: str, feature_scaler=None):
        """Save trained model with optional feature scaler"""
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        model_data = {
            'models': self.models,
            'n_components': self.n_components,
            'random_state': self.random_state,
            'trained': self.trained,
            'feature_scaler': self.feature_scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load trained model and return feature scaler if available"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.n_components = model_data['n_components']
        self.random_state = model_data['random_state']
        self.trained = model_data['trained']
        self.feature_scaler = model_data.get('feature_scaler', None)





