import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional

from expressivenote import ExpressiveNote  


class FMExpressiveModel:
    """Simple Flow Matching model for predicting expressive parameters"""
    
    def __init__(self, context_dim: int = 6, expression_dim: int = 4, hidden_dim: int = 128):
        self.context_dim = context_dim
        self.expression_dim = expression_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vector field network
        self.net = nn.Sequential(
            nn.Linear(context_dim + expression_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expression_dim)
        ).to(self.device)
        
        # Scalers for normalization
        self.context_scaler = StandardScaler()
        self.expression_scaler = StandardScaler()
        self.feature_encoders = {}
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
        self.trained = False
    
    def vector_field(self, t: torch.Tensor, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Vector field v_t(x) that defines the flow
        
        Args:
            t: time tensor [batch_size, 1] 
            x: current state [batch_size, expression_dim]
            context: musical context [batch_size, context_dim]
        """
        batch_size = x.shape[0]
        t_expanded = t.expand(batch_size, 1)
        input_vec = torch.cat([x, context, t_expanded], dim=1)
        return self.net(input_vec)
    
    def flow_matching_loss(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss
        
        Args:
            context: musical context features [batch_size, context_dim]
            target: target expression parameters [batch_size, expression_dim]
        """
        batch_size = context.shape[0]
        
        # Sample random time t ∈ [0, 1]
        t = torch.rand(batch_size, 1, device=self.device)
        
        # Sample noise z₀ ~ N(0, I)
        noise = torch.randn_like(target)
        
        # Linear interpolation: x_t = t * x₁ + (1-t) * x₀
        x_t = t * target + (1 - t) * noise
        
        # True velocity field: dx/dt = x₁ - x₀
        true_velocity = target - noise
        
        # Predicted velocity field
        pred_velocity = self.vector_field(t, x_t, context)
        
        # MSE loss
        return F.mse_loss(pred_velocity, true_velocity)
    
    def _encode_categorical_features(self, notes: List[ExpressiveNote], fit: bool = False) -> torch.Tensor:
        """Encode categorical features to numerical (same as before)"""
        categorical_features = []
        continuous_features = []
        
        for note in notes:
            categorical_features.append([note.rhythmic_context, note.ir_label])
            continuous_features.append([
                note.pitch_interval,
                note.duration_ratio,
                note.ir_closure,
                note.position_in_phrase
            ])
        
        # Encode categorical features
        if fit:
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
        return torch.tensor(all_features, dtype=torch.float32)
    
    def train(self, training_notes: List[List[ExpressiveNote]], epochs: int = 1000, batch_size: int = 32):
        """Train the flow matching model"""
        print("Training Flow Matching model...")
        
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
        
        # Extract and normalize features
        context_features = self._encode_categorical_features(training_notes_filtered, fit=True)
        context_features = self.context_scaler.fit_transform(context_features.numpy())
        context_features = torch.tensor(context_features, dtype=torch.float32, device=self.device)
        
        # Extract and normalize targets
        targets = np.array([[
            note.beat_period,
            note.timing, 
            note.velocity / 127.0,  # Normalize velocity to [0,1]
            note.articulation_log
        ] for note in training_notes_filtered])
        
        targets = self.expression_scaler.fit_transform(targets)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # Training loop
        dataset_size = len(context_features)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(dataset_size)
            context_shuffled = context_features[perm]
            targets_shuffled = targets[perm]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                
                batch_context = context_shuffled[start_idx:end_idx]
                batch_targets = targets_shuffled[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                loss = self.flow_matching_loss(batch_context, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.trained = True
        print("Training completed!")
    
    def sample(self, context: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Sample from the flow model using Euler integration
        
        Args:
            context: musical context [batch_size, context_dim]
            n_steps: number of integration steps
        """
        self.net.eval()
        with torch.no_grad():
            batch_size = context.shape[0]
            
            # Start from noise x₀ ~ N(0, I)
            x = torch.randn(batch_size, self.expression_dim, device=self.device)
            
            # Integration step size
            dt = 1.0 / n_steps
            
            # Euler integration from t=0 to t=1
            for step in range(n_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                dx_dt = self.vector_field(t, x, context)
                x = x + dt * dx_dt
            
            return x
    
    def predict(self, notes: List[ExpressiveNote]) -> List[ExpressiveNote]:
        """Predict expressive parameters for new notes"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract and normalize features
        context_features = self._encode_categorical_features(notes, fit=False)
        context_features = self.context_scaler.transform(context_features.numpy())
        context_features = torch.tensor(context_features, dtype=torch.float32, device=self.device)
        
        # Sample from flow
        predictions = self.sample(context_features)
        
        # Denormalize predictions
        predictions = predictions.cpu().numpy()
        predictions = self.expression_scaler.inverse_transform(predictions)
        
        # Create predicted notes
        predicted_notes = []
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
                position_in_phrase=note.position_in_phrase,
                beat_period=float(np.clip(predictions[i, 0], 0.3, 3.0)),
                timing=float(np.clip(predictions[i, 1], -0.5, 0.5)),
                velocity=int(np.clip(predictions[i, 2] * 127, 1, 127)),
                articulation_log=float(np.clip(predictions[i, 3], -2.0, 1.0))
            )
            predicted_notes.append(new_note)
        
        return predicted_notes
    
    def save(self, filepath: str):
        """Save trained model"""
        model_data = {
            'net_state_dict': self.net.state_dict(),
            'context_scaler': self.context_scaler,
            'expression_scaler': self.expression_scaler,
            'feature_encoders': self.feature_encoders,
            'context_dim': self.context_dim,
            'expression_dim': self.expression_dim,
            'hidden_dim': self.hidden_dim,
            'trained': self.trained
        }
        torch.save(model_data, filepath)
    
    def load(self, filepath: str):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Recreate network with saved dimensions
        self.context_dim = model_data['context_dim']
        self.expression_dim = model_data['expression_dim'] 
        self.hidden_dim = model_data['hidden_dim']
        
        self.net = nn.Sequential(
            nn.Linear(self.context_dim + self.expression_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.expression_dim)
        ).to(self.device)
        
        # Load state
        self.net.load_state_dict(model_data['net_state_dict'])
        self.context_scaler = model_data['context_scaler']
        self.expression_scaler = model_data['expression_scaler']
        self.feature_encoders = model_data['feature_encoders']
        self.trained = model_data['trained']
