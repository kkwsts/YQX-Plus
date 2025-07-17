"""
XGBoost-based expressive performance model

Based on the MidiHum XGBoost implementation for velocity prediction,
extended to predict all expressive parameters (timing, dynamics, articulation).
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
# import wandb
from torchinfo import summary


class XGBoostExpressiveModel:
    """
    XGBoost-based expressive performance model
    Compatible with YQX system interface
    
    Predicts expressive performance parameters (beat_period, timing, velocity, articulation_log)
    from musical context features using gradient boosted trees.
    """
    
    def __init__(self, 
                 features_dim: int = None,
                 target_dim: int = 4,
                 use_midihum: bool = False,
                 learning_rate: float = 0.05,
                 max_depth: int = 7,
                 n_estimators: int = 1000,
                 subsample: float = 0.9,
                 colsample_bytree: float = 0.6,
                 reg_alpha: float = 0.2,
                 reg_lambda: float = 0.4,
                 n_jobs: int = -1,
                 device: str = 'cpu',
                 tree_method: str = 'auto'):
        
        self.features_dim = features_dim
        self.target_dim = target_dim
        self.use_midihum = use_midihum
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.device = device
        self.tree_method = tree_method
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Initialize models 
        self.models = []
        self.trained = False
        
        # Target parameter names
        self.target_names = ['beat_period', 'timing', 'velocity', 'articulation_log']
        
        print(f"Initialized XGBoost model with {target_dim} targets")
        print(f"Target parameters: {self.target_names}")
    
    def _create_xgb_model(self) -> xgb.XGBRegressor:
        """Create an XGBoost regressor with configured parameters"""
        return xgb.XGBRegressor(
            booster="gbtree",
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            gamma=0.1,
            min_child_weight=7,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            device=self.device,
            enable_categorical=True 
        )
    
    def train(self, context_features: np.ndarray, targets: np.ndarray,
              val_features: np.ndarray = None, val_targets: np.ndarray = None,
              epochs: int = None, batch_size: int = None, patience: int = None):
        """
        Train the XGBoost model on pre-extracted features and targets
        
        Args:
            context_features: Input features [n_samples, features_dim]
            targets: Target parameters [n_samples, target_dim]
            val_features: Validation features (optional)
            val_targets: Validation targets (optional)
            epochs, batch_size, patience are kept for interface compatibility
        """
        print("Training XGBoost expressive model...")
        print(f"Training on {len(context_features)} samples")
        
        # Set features_dim if not already set
        if self.features_dim is None:
            self.features_dim = context_features.shape[1]
            print(f"Set features_dim to {self.features_dim}")
        
        # Scale features and targets
        context_features_scaled = self.feature_scaler.fit_transform(context_features)
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        # Prepare validation data if provided
        val_context_features_scaled = None
        val_targets_scaled = None
        if val_features is not None and val_targets is not None:
            print(f"Using validation set with {len(val_features)} samples")
            val_context_features_scaled = self.feature_scaler.transform(val_features)
            val_targets_scaled = self.target_scaler.transform(val_targets)
        
        # Train separate model for each target parameter
        self.models = []
        train_scores = []
        val_scores = []
        
        for i, target_name in enumerate(self.target_names):
            print(f"\nTraining model for {target_name}...")
            
            # Create and train model for this target
            model = self._create_xgb_model()
            
            # Train the model
            model.fit(
                context_features_scaled, 
                targets_scaled[:, i],
                verbose=False
            )
            
            # Evaluate on training set
            train_pred = model.predict(context_features_scaled)
            train_score = model.score(context_features_scaled, targets_scaled[:, i])
            train_scores.append(train_score)
            
            # Evaluate on validation set if available
            val_score = None
            if val_context_features_scaled is not None:
                val_pred = model.predict(val_context_features_scaled)
                val_score = model.score(val_context_features_scaled, val_targets_scaled[:, i])
                val_scores.append(val_score)
            
            self.models.append(model)
            
            print(f"  {target_name}: Train R² = {train_score:.4f}", end="")
            if val_score is not None:
                print(f", Val R² = {val_score:.4f}")
            else:
                print()
            
            # Log to wandb
            # wandb.log({
            #     f"xgboost_{target_name}_train_r2": train_score,
            #     f"xgboost_{target_name}_val_r2": val_score if val_score is not None else 0.0
            # })
        
        # Calculate overall scores
        avg_train_score = np.mean(train_scores)
        avg_val_score = np.mean(val_scores) if val_scores else None
        
        print(f"\nOverall Training R²: {avg_train_score:.4f}")
        if avg_val_score is not None:
            print(f"Overall Validation R²: {avg_val_score:.4f}")
        
        # wandb.log({
        #     "xgboost_avg_train_r2": avg_train_score,
        #     "xgboost_avg_val_r2": avg_val_score if avg_val_score is not None else 0.0
        # })
        
        self.trained = True
        print("XGBoost training completed!")
    
    def predict(self, context_features: np.ndarray) -> np.ndarray:
        """
        Predict expressive parameters using trained XGBoost models
        
        Args:
            context_features: Input features [n_samples, features_dim]
            
        Returns:
            Predicted targets [n_samples, target_dim]
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Scale features
        context_features_scaled = self.feature_scaler.transform(context_features)
        
        # Make predictions for each target parameter
        predictions_scaled = np.zeros((len(context_features_scaled), self.target_dim))
        
        for i, model in enumerate(self.models):
            predictions_scaled[:, i] = model.predict(context_features_scaled)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance scores for each target parameter
        
        Returns:
            Dictionary mapping target names to feature importance arrays
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict = {}
        
        for i, (target_name, model) in enumerate(zip(self.target_names, self.models)):
            importance = model.feature_importances_
            importance_dict[target_name] = importance
        
        return importance_dict
    
    def save(self, filepath: str, feature_scaler=None, save_best: bool = True):
        """
        Args:
            filepath: Path to save the model
            feature_scaler: Feature scaler
            (save_best is kept for interface compatibility)
        """
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'features_dim': self.features_dim,
            'target_dim': self.target_dim,
            'target_names': self.target_names,
            'use_midihum': self.use_midihum,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'n_jobs': self.n_jobs,
            'device': self.device,
            'tree_method': self.tree_method,
            'trained': self.trained,
            'model_type': 'last'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"XGBoost model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.features_dim = model_data['features_dim']
        self.target_dim = model_data['target_dim']
        self.target_names = model_data['target_names']
        self.use_midihum = model_data['use_midihum']
        self.learning_rate = model_data['learning_rate']
        self.max_depth = model_data['max_depth']
        self.n_estimators = model_data['n_estimators']
        self.subsample = model_data['subsample']
        self.colsample_bytree = model_data['colsample_bytree']
        self.reg_alpha = model_data['reg_alpha']
        self.reg_lambda = model_data['reg_lambda']
        self.n_jobs = model_data['n_jobs']
        self.device = model_data.get('device', 'cpu')
        self.tree_method = model_data.get('tree_method', 'auto') 
        self.trained = model_data['trained']
        
        model_type = model_data.get('model_type', 'unknown')
        if model_type == 'last':
            print("Loaded XGBoost model (final training)")
        else:
            print("Loaded XGBoost model (legacy format)")
        
        print(f"Loaded {len(self.models)} XGBoost models for targets: {self.target_names}")
    