import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional

from expressivenote import *  


class ConditionalFlowMatcher:
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        t = self._pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = self._pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        
        eps = torch.randn_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def _pad_t_like_x(self, t, x):
        if isinstance(t, (float, int)):
            return t
        return t.reshape(-1, *([1] * (x.dim() - 1)))



class FMExpressiveModel:
    """Flow Matching model for predicting expressive parameters"""
    
    def __init__(self, context_dim: int = 9, expression_dim: int = 4, hidden_dim: int = 128, 
                 use_midihum: bool = False, flow_matcher_type: str = "standard", sigma: float = 0.01):
        
        self.context_dim = context_dim  # Total feature dimension from features.py
        self.expression_dim = expression_dim
        self.hidden_dim = hidden_dim
        self.use_midihum = use_midihum
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.flow_matcher_type = flow_matcher_type
        if flow_matcher_type == "standard":
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(f"Unsupported flow matcher type: {flow_matcher_type}")
        
        input_dim = context_dim + expression_dim + 1
        # Vector field network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, expression_dim)
        ).to(self.device)
        
        # # Scalers for normalization
        self.context_scaler = StandardScaler()
        self.expression_scaler = StandardScaler()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        self.trained = False
    
    def vector_field(self, t: torch.Tensor, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Vector field v_t(x) that defines the flow
        
        Args:
            t: time tensor [batch_size, 1] 
            x: current state [batch_size, expression_dim]
            context: musical context features [batch_size, context_dim]
        """
        batch_size = x.shape[0]
        if t.dim() == 1: 
            t_expanded = t.unsqueeze(1)
        elif t.dim() == 2:
            if t.shape[1] == 1:
                t_expanded = t
            else:
                t_expanded = t.reshape(batch_size, 1)
        # t_expanded = t.expand(batch_size, 1)
        # t_expanded = t.unsqueeze(1)
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

        noise = torch.randn_like(target)
             
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, target)
        t = t.to(self.device)
        xt = xt.to(self.device)
        ut = ut.to(self.device)
        
        context_expanded = context.repeat_interleave(t.shape[0] // context.shape[0], dim=0)
        
        pred_velocity = self.vector_field(t, xt, context_expanded)
        
        return F.mse_loss(pred_velocity, ut)
    
    def train(self, training_notes: List[List[ExpressiveNote]], feature_extractor, epochs: int = 1000, batch_size: int = 32):
        """Train the flow matching model using features.py"""
        print("Training Flow Matching model with features.py...")
        
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
        
        context_features = feature_extractor.encode_features(training_notes_filtered, fit=True, use_midihum=self.use_midihum)
        context_features = self.context_scaler.fit_transform(context_features)
        context_features = torch.tensor(context_features, dtype=torch.float32, device=self.device)
        
        print(f"Context features shape: {context_features.shape}")  # (num_samples, context_dim)
        assert context_features.shape[1] == self.context_dim, f"Expected context_dim={self.context_dim}, but got {context_features.shape[1]}"

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
            
            x = torch.randn(batch_size, self.expression_dim, device=self.device)
            
            dt = 1.0 / n_steps
            
            # Euler integration from t=0 to t=1
            for step in range(n_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                dx_dt = self.vector_field(t, x, context)
                x = x + dt * dx_dt
            
            return x
    
    def predict(self, notes: List[ExpressiveNote], feature_extractor) -> List[ExpressiveNote]:
        """Predict expressive parameters for new notes using features.py"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        context_features = feature_extractor.encode_features(notes, fit=False, use_midihum=self.use_midihum)
        context_features = self.context_scaler.transform(context_features)
        context_features = torch.tensor(context_features, dtype=torch.float32, device=self.device)
        
        predictions = self.sample(context_features)
        
        predictions = predictions.cpu().numpy()
        predictions = self.expression_scaler.inverse_transform(predictions)
        
        predicted_notes = []
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
            'context_dim': self.context_dim,
            'expression_dim': self.expression_dim,
            'hidden_dim': self.hidden_dim,
            'use_midihum': self.use_midihum,
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
        self.use_midihum = model_data.get('use_midihum', True)
        input_dim = self.context_dim + self.expression_dim + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.expression_dim)
        ).to(self.device)
        
        # Load state
        self.net.load_state_dict(model_data['net_state_dict'])
        self.context_scaler = model_data['context_scaler']
        self.expression_scaler = model_data['expression_scaler']
        self.trained = model_data['trained']

    def test_with_features(self, feature_extractor):
        """Test the model with features.py to verify compatibility"""
        print("Testing FMExpressiveModel with features.py...")
        
        # Create some dummy ExpressiveNote objects
        test_notes = [
            ExpressiveNote(
                pitch=60, onset_beat=0.0, duration_beat=1.0, voice=0,
                pitch_interval=2, duration_ratio=1.0, rhythmic_context="s-s-l",
                ir_label="Process", ir_closure=0.1, position_in_phrase=0.5,
                beat_period=0.5, timing=0.1, velocity=64, articulation_log=0.0
            ),
            ExpressiveNote(
                pitch=62, onset_beat=1.0, duration_beat=0.5, voice=0,
                pitch_interval=-1, duration_ratio=0.5, rhythmic_context="l-s-s",
                ir_label="Reversal", ir_closure=0.3, position_in_phrase=0.8,
                beat_period=0.6, timing=-0.05, velocity=72, articulation_log=0.2
            )
        ]
        
        # Test feature encoding
        try:
            encoded_features = feature_extractor.encode_features(test_notes, fit=True, use_midihum=self.use_midihum)
            print(f"✓ Feature encoding successful. Shape: {encoded_features.shape}")
            print(f"✓ Context dimension matches: {encoded_features.shape[1]} == {self.context_dim}")
            
            # Test model forward pass
            context_tensor = torch.tensor(encoded_features, dtype=torch.float32, device=self.device)
            t = torch.rand(2, 1, device=self.device)
            x = torch.randn(2, self.expression_dim, device=self.device)
            
            output = self.vector_field(t, x, context_tensor)
            print(f"✓ Model forward pass successful. Output shape: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False
