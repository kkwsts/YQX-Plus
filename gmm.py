import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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
        self.validation_scores = {}  # Store validation performance
        self.best_n_components = {}  # Store best n_components per target
    
    def _compute_validation_score(self, model: GaussianMixture, 
                                 val_features: np.ndarray, val_target: np.ndarray) -> float:
        """Compute validation score for a GMM model"""
        try:
            # Predict on validation set
            predictions = np.zeros(len(val_features))
            
            for i in range(len(val_features)):
                features = val_features[i:i+1]
                
                # Find most likely component
                log_probs = []
                for comp in range(model.n_components):
                    mean = model.means_[comp]
                    cov = model.covariances_[comp]
                    
                    # Compute log probability of features under this component
                    feature_mean = mean[:-1]
                    feature_cov = cov[:-1, :-1]
                    
                    try:
                        inv_cov = np.linalg.inv(feature_cov)
                        diff = features[0] - feature_mean
                        log_prob = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
                        log_probs.append(log_prob + np.log(model.weights_[comp]))
                    except:
                        log_probs.append(-np.inf)
                
                best_comp = np.argmax(log_probs)
                target_mean = model.means_[best_comp, -1]
                predictions[i] = target_mean
            
            # Compute MSE
            mse = mean_squared_error(val_target, predictions)
            return mse
        except:
            return float('inf')
    
    def _select_best_n_components(self, context_features: np.ndarray, targets: np.ndarray,
                                 val_features: np.ndarray, val_targets: np.ndarray,
                                 target_name: str) -> Tuple[int, GaussianMixture]:
        """Select best number of components using validation data"""
        print(f"Selecting best n_components for {target_name}...")
        
        best_score = float('inf')
        best_model = None
        best_n = self.n_components
        
        # Test different numbers of components
        n_components_range = range(2, min(self.n_components + 1, len(context_features) // 10))
        
        for n_comp in n_components_range:
            try:
                model = GaussianMixture(n_components=n_comp, random_state=self.random_state)
                joint_data = np.column_stack([context_features, targets])
                model.fit(joint_data)
                
                # Compute validation score
                val_score = self._compute_validation_score(model, val_features, val_targets)
                
                print(f"  n_components={n_comp}: validation MSE = {val_score:.6f}")
                
                if val_score < best_score:
                    best_score = val_score
                    best_model = model
                    best_n = n_comp
                    
            except Exception as e:
                print(f"  n_components={n_comp}: failed to fit - {e}")
                continue
        
        print(f"Best n_components for {target_name}: {best_n} (MSE: {best_score:.6f})")
        return best_n, best_model
    
    def train(self, context_features: np.ndarray, targets: np.ndarray, 
              val_features: np.ndarray = None, val_targets: np.ndarray = None, **kwargs):
        """Train the Bayesian model on pre-extracted features and targets"""
        print("Training GMM model...")
        print(f"Training on {len(context_features)} samples")
        if val_features is not None:
            print(f"Validation set: {len(val_features)} samples")
        
        # Split targets into separate arrays
        target_dict = {
            'beat_period': targets[:, 0],
            'timing': targets[:, 1], 
            'velocity': targets[:, 2],
            'articulation_log': targets[:, 3]
        }
        
        # Split validation targets if available
        val_target_dict = None
        if val_targets is not None:
            val_target_dict = {
                'beat_period': val_targets[:, 0],
                'timing': val_targets[:, 1], 
                'velocity': val_targets[:, 2],
                'articulation_log': val_targets[:, 3]
            }
        
        # Train separate models for each target
        for target_name, y in target_dict.items():
            print(f"Training {target_name} model...")
            
            if (val_features is not None and val_target_dict is not None and len(val_features) > 0):
                # Use validation data to select best n_components
                best_n, best_model = self._select_best_n_components(
                    context_features, y, val_features, val_target_dict[target_name], target_name
                )
                self.best_n_components[target_name] = best_n
                self.models[target_name] = best_model
                
                # Store validation score
                val_score = self._compute_validation_score(
                    best_model, val_features, val_target_dict[target_name]
                )
                self.validation_scores[target_name] = val_score
                
            else:
                print(f"Using n_components={self.n_components} for {target_name}")
                model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
                joint_data = np.column_stack([context_features, y])
                model.fit(joint_data)
                self.models[target_name] = model
                
                if val_features is not None and val_target_dict is not None:
                    val_score = self._compute_validation_score(
                        model, val_features, val_target_dict[target_name]
                    )
                    self.validation_scores[target_name] = val_score
                    print(f"Validation MSE for {target_name}: {val_score:.6f}")
        
        self.trained = True
        print("GMM training completed!")
        
        # Print summary of validation performance
        if self.validation_scores:
            print("\nValidation Performance Summary:")
            for target_name, score in self.validation_scores.items():
                n_comp = self.best_n_components.get(target_name, self.n_components)
                print(f"  {target_name}: MSE={score:.6f}, n_components={n_comp}")
    
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
                for comp in range(model.n_components):
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
    
    def save(self, filepath: str, feature_scaler=None, save_best: bool = True):
        """Save trained model with optional feature scaler"""
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        model_data = {
            'models': self.models,
            'n_components': self.n_components,
            'random_state': self.random_state,
            'trained': self.trained,
            'feature_scaler': self.feature_scaler,
            'validation_scores': self.validation_scores,
            'best_n_components': self.best_n_components,
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
        self.validation_scores = model_data.get('validation_scores', {})
        self.best_n_components = model_data.get('best_n_components', {})





