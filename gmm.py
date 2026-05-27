import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
import torch
from gmm_gpu.gmm import GMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# import hook
import warnings

from expressivenote import ExpressiveNote

# Configure linear algebra backend to avoid CUDA solver issues
torch.backends.cuda.preferred_linalg_library('magma')

class BayesianExpressiveModel:
    """Bayesian model for predicting expressive parameters using gmm-gpu"""
    
    def __init__(self, n_components: int = 8, random_state: int = 42, 
                 use_gpu: bool = True, batch_size: int = 1024, **trainer_params):
        self.n_components = n_components
        self.random_state = random_state
        self.models = {}  # One model per target variable
        self.trained = False
        self.feature_scaler = None  # Store scaler in model
        self.validation_scores = {}  # Store validation performance
        self.best_n_components = {}  # Store best n_components per target
        
        # gmm-gpu specific parameters
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Multi-round training parameters to compensate for smaller sample size
        self.n_training_rounds = 3  # Number of training rounds with different samples
        self.samples_per_round = 13000  # Samples per round (reduced from 100k)
        
        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _convert_to_torch(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return data
    
    def _convert_to_numpy(self, data: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def _compute_validation_score(self, model: GMM, 
                                 val_features: np.ndarray, val_target: np.ndarray) -> float:
        """Compute validation score for a gmm-gpu model"""
        # Convert to torch tensors
        val_features_torch = self._convert_to_torch(val_features)
        val_target_torch = self._convert_to_torch(val_target)
        
        # Move to device if using GPU
        if torch.cuda.is_available():
            val_features_torch = val_features_torch.to(self.device)
            val_target_torch = val_target_torch.to(self.device)
        
        # Get model parameters
        means = model.means  # This is a list of tensors
        covariances = model.covs
        weights = model._pi
        
        # Check the expected batch size from the model
        expected_batch_size = means[0].shape[0] if means else len(val_features)
        
        # If validation set is larger than expected batch size, sample it
        if len(val_features) > expected_batch_size:
            print(f"Validation set too large ({len(val_features)} > {expected_batch_size}). Sampling...")
            np.random.seed(42)
            indices = np.random.choice(len(val_features), expected_batch_size, replace=False)
            val_features_torch = val_features_torch[indices]
            val_target_torch = val_target_torch[indices]
            val_features = val_features[indices]
            val_target = val_target[indices]
            print(f"Sampled validation set to {len(val_features)} samples")
        
        # Predict on validation set
        predictions = np.zeros(len(val_features))
        
        for i in range(len(val_features)):
            features = val_features_torch[i:i+1]
            
            # Find most likely component
            log_probs = []
            for comp in range(model._n_components):
                mean = means[comp]  # Shape: (batch_size, features)
                cov = covariances[comp]
                
                # For this specific sample i, get the mean and cov
                sample_mean = mean[i]  # Shape: (features,)
                sample_cov = cov[i]    # Shape: (features, features)
                
                # Compute log probability of features under this component
                feature_mean = sample_mean[:-1]  # All but last dimension (target)
                feature_cov = sample_cov[:-1, :-1]
                
                try:
                    # Use torch operations for better GPU compatibility
                    inv_cov = torch.inverse(feature_cov)
                    diff = features[0] - feature_mean
                    log_prob = -0.5 * torch.dot(diff, torch.matmul(inv_cov, diff))
                    log_probs.append(log_prob + torch.log(weights[comp]))
                except:
                    log_probs.append(torch.tensor(-float('inf')))
            
            best_comp = torch.argmax(torch.stack(log_probs))
            target_mean = means[best_comp][i, -1]  # Get target value for this sample
            predictions[i] = self._convert_to_numpy(target_mean)
        
        # remove nan values and keep shape the same as val_target
        nan_mask = np.isnan(predictions)
        predictions = predictions[~nan_mask]
        val_target = val_target[~nan_mask]
        
        mse = mean_squared_error(val_target, predictions)
        return mse


    
    def _select_best_n_components(self, context_features: np.ndarray, targets: np.ndarray,
                                 val_features: np.ndarray, val_targets: np.ndarray,
                                 target_name: str) -> Tuple[int, GMM]:
        """Select best number of components using validation data"""
        print(f"Selecting best n_components for {target_name}...")
        
        best_score = float('inf')
        best_model = None
        best_n = self.n_components
        
        # Test different numbers of components
        n_components_range = range(2, min(self.n_components + 1, len(context_features) // 10))
        
        for n_comp in n_components_range:
            # gmm_gpu initialises with modified kmeans++ refined by kmeans
            # (matches the paper's "K-Means initialization" claim). Passing
            # random_seed makes the fit reproducible across runs.
            model = GMM(
                n_components=n_comp,
                device=self.device,
                random_seed=self.random_state,
            )
            
            # Prepare joint data - reshape to 3D format expected by gmm-gpu
            joint_data = np.column_stack([context_features, targets])
            # Reshape to (batch_size, 1, features) where batch_size = number of samples
            joint_data_3d = joint_data.reshape(-1, 1, joint_data.shape[1])
            joint_data_torch = self._convert_to_torch(joint_data_3d)
            
            # Comprehensive NaN handling
            # First, check if there are any NaN values
            if torch.isnan(joint_data_torch).any():
                print(f"Warning: Found NaN values in {target_name} data. Cleaning...")
                # Remove rows with any NaN values
                nan_mask = torch.isnan(joint_data_torch).any(dim=(1, 2))
                joint_data_torch = joint_data_torch[~nan_mask]
                print(f"Removed {nan_mask.sum().item()} rows with NaN values. Remaining: {joint_data_torch.shape[0]}")
                
                # If still have NaN values, replace them with zeros
                if torch.isnan(joint_data_torch).any():
                    print("Warning: Still have NaN values after row removal. Replacing with zeros...")
                    joint_data_torch = torch.nan_to_num(joint_data_torch, nan=0.0)
            
            # Ensure data is finite (no inf values)
            if not torch.isfinite(joint_data_torch).all():
                print(f"Warning: Found infinite values in {target_name} data. Replacing with zeros...")
                joint_data_torch = torch.where(torch.isfinite(joint_data_torch), joint_data_torch, torch.zeros_like(joint_data_torch))
            
            # Check if we have enough data after cleaning
            if joint_data_torch.shape[0] < n_comp * 10:
                print(f"Warning: Not enough data for {n_comp} components. Skipping...")
                continue
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                joint_data_torch = joint_data_torch.to(self.device)
                print(f"Moved to device: {self.device}")
            
            # Fit the model
            try:
                model.fit(joint_data_torch)
            except torch._C._LinAlgError as e:
                print(f"CUDA linear algebra error for {target_name} with {n_comp} components. Trying CPU fallback...")
                # Try with CPU instead
                model_cpu = GMM(
                    n_components=n_comp,
                    device='cpu',
                    random_seed=self.random_state,
                )
                joint_data_cpu = joint_data_torch.cpu()
                try:
                    model_cpu.fit(joint_data_cpu)
                    model = model_cpu
                    print(f"Successfully fitted {target_name} model on CPU with {n_comp} components")
                except Exception as e2:
                    print(f"CPU fitting also failed for {target_name} with {n_comp} components: {e2}")
                    continue
            except Exception as e:
                print(f"Fitting failed for {target_name} with {n_comp} components: {e}")
                continue
            
            # Compute validation score
            val_score = self._compute_validation_score(model, val_features, val_targets)
            
            print(f"  n_components={n_comp}: validation MSE = {val_score:.6f}")
            
            if val_score < best_score:
                best_score = val_score
                best_model = model
                best_n = n_comp

            
        
        print(f"Best n_components for {target_name}: {best_n} (MSE: {best_score:.6f})")
        return best_n, best_model
    
    def _train_with_batching(self, context_features: np.ndarray, targets: np.ndarray, 
                           val_features: np.ndarray = None, val_targets: np.ndarray = None, 
                           target_name: str = "target"):
        """Train GMM model using batched processing to handle large datasets"""
        print(f"Training {target_name} with batched processing...")
        
        # Determine optimal batch size based on available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory > 40:  # Large GPU
                batch_size = min(13000, len(context_features) // 4)
            elif gpu_memory > 20:  # Medium GPU
                batch_size = min(25000, len(context_features) // 8)
            else:  # Small GPU
                batch_size = min(13000, len(context_features) // 16)
        else:
            batch_size = min(13000, len(context_features) // 16)
        
        print(f"Using batch size: {batch_size}")
        
        # Sample data for training to reduce memory usage
        if len(context_features) > batch_size * 4:
            np.random.seed(42)
            indices = np.random.choice(len(context_features), batch_size * 4, replace=False)
            context_features = context_features[indices]
            targets = targets[indices]
            print(f"Sampled {len(context_features)} samples for training")
        
        # Create GMM model
        model = GMM(n_components=self.n_components, device=self.device, random_seed=self.random_state)
        
        # Prepare data
        joint_data = np.column_stack([context_features, targets])
        joint_data_3d = joint_data.reshape(-1, 1, joint_data.shape[1])
        joint_data_torch = self._convert_to_torch(joint_data_3d)
        
        # Clean data
        if torch.isnan(joint_data_torch).any():
            nan_mask = torch.isnan(joint_data_torch).any(dim=(1, 2))
            joint_data_torch = joint_data_torch[~nan_mask]
        
        if not torch.isfinite(joint_data_torch).all():
            joint_data_torch = torch.where(torch.isfinite(joint_data_torch), 
                                          joint_data_torch, torch.zeros_like(joint_data_torch))
        
        # Move to device
        if torch.cuda.is_available():
            joint_data_torch = joint_data_torch.to(self.device)
        
        # Fit model with error handling
        try:
            model.fit(joint_data_torch)
            return model
        except torch._C._LinAlgError as e:
            print(f"CUDA error for {target_name}. Trying CPU...")
            model_cpu = GMM(n_components=self.n_components, device='cpu', random_seed=self.random_state)
            joint_data_cpu = joint_data_torch.cpu()
            model_cpu.fit(joint_data_cpu)
            return model_cpu
        except Exception as e:
            print(f"Fitting failed for {target_name}: {e}")
            return None
    
    def _train_multi_round(self, context_features: np.ndarray, targets: np.ndarray, 
                          val_features: np.ndarray = None, val_targets: np.ndarray = None, 
                          target_name: str = "target"):
        """Train GMM model using multiple rounds with different data samples"""
        print(f"Training {target_name} with multi-round approach ({self.n_training_rounds} rounds)...")
        
        # Store all models from different rounds
        round_models = []
        round_scores = []
        
        for round_idx in range(self.n_training_rounds):
            print(f"  Round {round_idx + 1}/{self.n_training_rounds}")
            
            # Sample different data for this round
            np.random.seed(self.random_state + round_idx)  # Different seed for each round
            if len(context_features) > self.samples_per_round:
                indices = np.random.choice(len(context_features), self.samples_per_round, replace=False)
                round_features = context_features[indices]
                round_targets = targets[indices]
            else:
                round_features = context_features
                round_targets = targets
            
            # Create and train model for this round
            model = GMM(n_components=self.n_components, device=self.device, random_seed=self.random_state)
            
            # Prepare data
            joint_data = np.column_stack([round_features, round_targets])
            joint_data_3d = joint_data.reshape(-1, 1, joint_data.shape[1])
            joint_data_torch = self._convert_to_torch(joint_data_3d)
            
            # Clean data
            if torch.isnan(joint_data_torch).any():
                nan_mask = torch.isnan(joint_data_torch).any(dim=(1, 2))
                joint_data_torch = joint_data_torch[~nan_mask]
            
            if not torch.isfinite(joint_data_torch).all():
                joint_data_torch = torch.where(torch.isfinite(joint_data_torch), 
                                              joint_data_torch, torch.zeros_like(joint_data_torch))
            
            # Move to device
            if torch.cuda.is_available():
                joint_data_torch = joint_data_torch.to(self.device)
            
            # Fit model
            try:
                model.fit(joint_data_torch)
                round_models.append(model)
                
                # Compute validation score for this round
                if val_features is not None and val_targets is not None:
                    val_score = self._compute_validation_score(model, val_features, val_targets)
                    round_scores.append(val_score)
                    print(f"    Round {round_idx + 1} validation MSE: {val_score:.6f}")
                
                # Clean up GPU memory after each round
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
            except Exception as e:
                print(f"    Round {round_idx + 1} failed: {e}")
                # Clean up even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                continue
        
        if not round_models:
            print(f"All rounds failed for {target_name}")
            return None
        
        # Select best model based on validation score
        if round_scores:
            best_round = np.argmin(round_scores)
            best_model = round_models[best_round]
            best_score = round_scores[best_round]
            print(f"  Selected round {best_round + 1} with MSE: {best_score:.6f}")
        else:
            # If no validation scores, use the first successful model
            best_model = round_models[0]
            print(f"  Using first successful model (no validation scores)")
        
        return best_model
    
    def train(self, context_features: np.ndarray, targets: np.ndarray, 
              val_features: np.ndarray = None, val_targets: np.ndarray = None, **kwargs):
        """Train the Bayesian model on pre-extracted features and targets using gmm-gpu"""
        print("Training gmm-gpu GMM model...")
        
        print(f"Training on {len(context_features)} samples")
        if val_features is not None:
            print(f"Validation set: {len(val_features)} samples")
        
        # Limit dataset size to prevent GPU OOM
        max_samples = 13000  # Limit to 50k samples (reduced from 100k due to larger features)
        if len(context_features) > max_samples:
            print(f"Dataset too large ({len(context_features)} samples). Sampling {max_samples} samples...")
            # Random sampling
            np.random.seed(42)
            indices = np.random.choice(len(context_features), max_samples, replace=False)
            context_features = context_features[indices]
            targets = targets[indices]
            print(f"Sampled dataset: {len(context_features)} samples")
        
        # Also limit validation set if it's too large
        if val_features is not None and len(val_features) > max_samples:
            print(f"Validation set too large ({len(val_features)} samples). Sampling {max_samples} samples...")
            np.random.seed(42)
            val_indices = np.random.choice(len(val_features), max_samples, replace=False)
            val_features = val_features[val_indices]
            val_targets = val_targets[val_indices]
            print(f"Sampled validation set: {len(val_features)} samples")
        
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
            
            # Use multi-round training for large datasets to compensate for smaller sample size
            if len(context_features) > 100000:  # Use multi-round training for large datasets
                print(f"Large dataset detected ({len(context_features)} samples). Using multi-round training...")
                model = self._train_multi_round(context_features, y, val_features, 
                                              val_target_dict[target_name] if val_target_dict else None, 
                                              target_name)
                if model is not None:
                    self.models[target_name] = model
                    if val_features is not None and val_target_dict is not None:
                        val_score = self._compute_validation_score(
                            model, val_features, val_target_dict[target_name]
                        )
                        self.validation_scores[target_name] = val_score
                        print(f"Final validation MSE for {target_name}: {val_score:.6f}")
                continue
            
            # Skip component selection and use specified n_components directly for speed
            print(f"Using n_components={self.n_components} for {target_name} (skipping component selection)")
            
            model = GMM(
                n_components=self.n_components,
                device=self.device,
                random_seed=self.random_state,
            )
            
            # Prepare joint data - reshape to 3D format expected by gmm-gpu
            joint_data = np.column_stack([context_features, y])
            # Reshape to (batch_size, 1, features) where batch_size = number of samples
            joint_data_3d = joint_data.reshape(-1, 1, joint_data.shape[1])
            joint_data_torch = self._convert_to_torch(joint_data_3d)
            
            # Comprehensive NaN handling
            if torch.isnan(joint_data_torch).any():
                print(f"Warning: Found NaN values in {target_name} data. Cleaning...")
                nan_mask = torch.isnan(joint_data_torch).any(dim=(1, 2))
                joint_data_torch = joint_data_torch[~nan_mask]
                print(f"Removed {nan_mask.sum().item()} rows with NaN values. Remaining: {joint_data_torch.shape[0]}")
                
                if torch.isnan(joint_data_torch).any():
                    print("Warning: Still have NaN values after row removal. Replacing with zeros...")
                    joint_data_torch = torch.nan_to_num(joint_data_torch, nan=0.0)
            
            # Ensure data is finite
            if not torch.isfinite(joint_data_torch).all():
                print(f"Warning: Found infinite values in {target_name} data. Replacing with zeros...")
                joint_data_torch = torch.where(torch.isfinite(joint_data_torch), joint_data_torch, torch.zeros_like(joint_data_torch))
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                joint_data_torch = joint_data_torch.to(self.device)
            
            # Fit the model with error handling
            try:
                model.fit(joint_data_torch)
            except torch._C._LinAlgError as e:
                print(f"CUDA linear algebra error for {target_name}. Trying CPU fallback...")
                model_cpu = GMM(n_components=self.n_components, device='cpu', random_seed=self.random_state)
                joint_data_cpu = joint_data_torch.cpu()
                try:
                    model_cpu.fit(joint_data_cpu)
                    model = model_cpu
                    print(f"Successfully fitted {target_name} model on CPU")
                except Exception as e2:
                    print(f"CPU fitting also failed for {target_name}: {e2}")
                    continue
            except Exception as e:
                print(f"Fitting failed for {target_name}: {e}")
                continue
            
            self.models[target_name] = model
            
            # Compute validation score if validation data is available
            if val_features is not None and val_target_dict is not None:
                val_score = self._compute_validation_score(
                    model, val_features, val_target_dict[target_name]
                )
                self.validation_scores[target_name] = val_score
                print(f"Validation MSE for {target_name}: {val_score:.6f}")
            
            # Clean up GPU memory after each model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Monitor GPU memory
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Cleaned up GPU memory after {target_name} model. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        self.trained = True
        print("gmm-gpu GMM training completed!")
        
        # Print summary of validation performance
        if self.validation_scores:
            print("\nValidation Performance Summary:")
            for target_name, score in self.validation_scores.items():
                n_comp = self.best_n_components.get(target_name, self.n_components)
                print(f"  {target_name}: MSE={score:.6f}, n_components={n_comp}")
    
    def predict(self, context_features: np.ndarray) -> np.ndarray:
        """Predict targets for pre-extracted features using gmm-gpu models"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Initialize predictions array
        predictions = np.zeros((len(context_features), 4))
        
        # Convert features to torch tensor
        context_features_torch = self._convert_to_torch(context_features)
        
        # Move to device if using GPU
        if torch.cuda.is_available():
            context_features_torch = context_features_torch.to(self.device)
        
        # Predict each target variable
        target_names = ['beat_period', 'timing', 'velocity', 'articulation_log']
        for target_idx, target_name in enumerate(target_names):
            model = self.models[target_name]
            
            for i in range(len(context_features)):
                # Use conditional expectation given features
                # Approximate by finding most likely component and using its mean
                features = context_features_torch[i:i+1]
                
                # Get model parameters
                means = model.means  # This is a list of tensors
                covariances = model.covs
                weights = model._pi
                
                # Find most likely component
                log_probs = []
                for comp in range(model._n_components):
                    mean = means[comp]  # Shape: (batch_size, features)
                    cov = covariances[comp]
                    
                    # For this specific sample i, get the mean and cov
                    sample_mean = mean[i]  # Shape: (features,)
                    sample_cov = cov[i]    # Shape: (features, features)
                    
                    # Compute log probability of features under this component
                    feature_mean = sample_mean[:-1]  # All but last dimension (target)
                    feature_cov = sample_cov[:-1, :-1]
                    
                    try:
                        # Use torch operations for better GPU compatibility
                        inv_cov = torch.inverse(feature_cov)
                        diff = features[0] - feature_mean
                        log_prob = -0.5 * torch.dot(diff, torch.matmul(inv_cov, diff))
                        log_probs.append(log_prob + torch.log(weights[comp]))
                    except:
                        log_probs.append(torch.tensor(-float('inf')))
                
                best_comp = torch.argmax(torch.stack(log_probs))
                
                # Predict target using conditional mean
                target_mean = means[best_comp][i, -1]  # Get target value for this sample
                predictions[i, target_idx] = self._convert_to_numpy(target_mean)
        
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
            'use_gpu': self.use_gpu,
            'batch_size': self.batch_size,
            'device': self.device,
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
        self.use_gpu = model_data.get('use_gpu', True)
        self.batch_size = model_data.get('batch_size', 1024)
        self.device = model_data.get('device', 'cpu')





