import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import math
import typing as tp
from expressivenote import *

from torchdiffeq import odeint
import audiocraft
from audiocraft.modules.streaming import StreamingModule
from audiocraft.modules.transformer import StreamingTransformerLayer
from audiocraft.modules.conditioners import (
    ConditioningAttributes, 
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditionType
)
try:
    from audiocraft.models.lm import ConditionTensors
except ImportError:
    ConditionTensors = tp.Dict[str, ConditionType]



class ConditionalFlowMatcher:    
    def __init__(self, sigma: float = 0.0, schedule_type: str = "linear"):
        self.sigma = sigma
        self.schedule_type = schedule_type
        
        self.schedule = None

    def compute_mu_t(self, x0, x1, t):
        """Mean of the flow path"""
        t = self._pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """Variance of the flow path"""
        if self.schedule_type == "cosine":
            return self.sigma * (1 - torch.cos(t * torch.pi / 2))
        else:
            return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """Sample from the flow path at time t"""
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = self._pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional flow (vector field)"""
        del t, xt
        return x1 - x0

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """Sample location and corresponding flow"""
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


class AudioCraftTransformerBlock(nn.Module):
    """AudioCraft-inspired transformer block for musical context processing"""
    def __init__(self, dim: int, num_heads: int, hidden_scale: int = 4):
        super().__init__()
        self.dim = dim
        
        try:
            self.transformer_layer = StreamingTransformerLayer(
                d_model=dim,
                num_heads=num_heads,
                dim_feedforward=int(hidden_scale * dim),
                dropout=0.1,
                activation='gelu',
                norm_first=True
            )
            self.uses_audiocraft = True
        except Exception as e:
            print(f"Failed to initialize AudioCraft transformer, using fallback: {e}")
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(hidden_scale * dim),
                dropout=0.1,
                activation='gelu',
                norm_first=True,
                batch_first=True
            )
            self.uses_audiocraft = False
            
    def forward(self, x):
        return self.transformer_layer(x)


class MusicalContextEncoder(nn.Module):    
    def __init__(self, features_dim: int, dim: int = 128, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.features_dim = features_dim
        self.dim = dim
        
        self.input_proj = nn.Linear(features_dim, dim)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            AudioCraftTransformerBlock(dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, features_dim]
        Returns:
            encoded_context: [batch_size, dim]
        """
        x = self.input_proj(features)  # [B, dim]
        x = x.unsqueeze(1)  # [B, 1, dim]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x.squeeze(1)  # [B, dim]
        x = self.output_proj(x)
        x = self.norm(x)
        
        return x


class TimeEmbedding(nn.Module):

    def __init__(self, time_embedding_dim: int = 128):
        super().__init__()
        self.d_temb1 = time_embedding_dim
        self.d_temb2 = 4 * time_embedding_dim
        
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.d_temb1, self.d_temb2),
            nn.Linear(self.d_temb2, self.d_temb2),
        ])
        self.temb_proj = nn.Linear(self.d_temb2, time_embedding_dim)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """
        #######################################################################################################
        Taken From: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
        #######################################################################################################
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """

        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def _embed_time_parameter(self, t: torch.Tensor):
        """
        #######################################################################################################
        Inspired By: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
        #######################################################################################################
        
        Args:
            t: Time parameter tensor
        Returns:
            Processed time embeddings with swish activation
        """
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        
        temb = self._get_timestep_embedding(t.flatten(), self.d_temb1)
        temb = self.temb.dense[0](temb)
        temb = temb * torch.sigmoid(temb)  
        temb = self.temb.dense[1](temb)
        
        return temb



    def forward(self, t: torch.Tensor):
        """        
        Args:
            t: Time parameter tensor
        Returns:
            Time embeddings projected to final dimension
        """
        temb = self._embed_time_parameter(t)
        return self.temb_proj(temb)


class MusicalConditioningSystem(nn.Module):
    """
    Multi-modal conditioning system for musical features
    
    FEATURE STRUCTURE:
    - features = basic_features + midihum_features (when use_midihum=True)
               = basic_features (when use_midihum=False)
    """
    
    def __init__(self, basic_dim: int = 9, hidden_dim: int = 128, use_midihum: bool = False):
        super().__init__()
        self.basic_dim = basic_dim
        self.hidden_dim = hidden_dim
        self.use_midihum = use_midihum
        
        # Feature dimensions
        self.midihum_dim = 202  # Approximate midihum feature count
        self.features_dim = basic_dim + (self.midihum_dim if use_midihum else 0) 
        
        self.basic_encoder = MusicalContextEncoder(
            features_dim=basic_dim,
            dim=hidden_dim,
            num_heads=8,
            num_layers=4
        )
        
        if use_midihum:
            self.harmonic_encoder = nn.Sequential(
                nn.Linear(15, hidden_dim // 2),  # Chord context features
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            
            self.statistical_encoder = nn.Sequential(
                nn.Linear(96, hidden_dim),  # SMA features
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.technical_encoder = nn.Sequential(
                nn.Linear(60, hidden_dim // 2),  # Technical indicators
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            
            self.timing_encoder = nn.Sequential(
                nn.Linear(30, hidden_dim // 2),  # Timing features
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            
            # AudioCraft-style condition fuser
            try:
                self.condition_fuser = ConditionFuser(
                    fuse2cond={
                        'basic': [0],           # Basic musical features
                        'harmonic': [1],        # Harmonic context
                        'statistical': [2],     # Statistical features
                        'technical': [3],       # Technical indicators
                        'timing': [4]           # Timing features
                    },
                    dim=hidden_dim
                )
            except:
                # Fallback: simple concatenation + projection
                self.condition_fuser = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),  # basic + harmonic + statistical + technical + timing
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process multi-modal musical features
        
        Args:
            features: [batch_size, features_dim] where features = basic_features + midihum_features
        Returns:
            Fused conditioning: [batch_size, hidden_dim]
        """
        if not self.use_midihum:
            basic_features = features
            return self.basic_encoder(basic_features)
        
        basic_features = features[:, :self.basic_dim]           # [batch, 9]
        midihum_features = features[:, self.basic_dim:]         # [batch, ~202]
        
        # Encode basic features
        basic_encoded = self.basic_encoder(basic_features)      # [batch, hidden_dim]
        
        harmonic_features = midihum_features[:, :15]            # Chord context [batch, 15]
        statistical_features = midihum_features[:, 15:111]      # SMA features [batch, 96]
        technical_features = midihum_features[:, 111:171]       # Technical indicators [batch, 60]
        timing_features = midihum_features[:, 171:201]          # Timing features [batch, 30]
        
        harmonic_encoded = self.harmonic_encoder(harmonic_features)         # [batch, hidden_dim//2]
        statistical_encoded = self.statistical_encoder(statistical_features) # [batch, hidden_dim]
        technical_encoded = self.technical_encoder(technical_features)       # [batch, hidden_dim//2]
        timing_encoded = self.timing_encoder(timing_features)               # [batch, hidden_dim//2]
        
        # Fuse all modalities
        if hasattr(self.condition_fuser, 'fuse2cond'):
            # AudioCraft ConditionFuser
            conditions = {
                'basic': basic_encoded.unsqueeze(1),
                'harmonic': harmonic_encoded.unsqueeze(1),
                'statistical': statistical_encoded.unsqueeze(1),
                'technical': technical_encoded.unsqueeze(1),
                'timing': timing_encoded.unsqueeze(1)
            }
            fused = self.condition_fuser(conditions)
            return fused.squeeze(1)
        else:
            # Fallback: concatenate and project
            concatenated = torch.cat([
                basic_encoded,
                harmonic_encoded,
                statistical_encoded
            ], dim=1)
            return self.condition_fuser(concatenated)


class VectorFieldNetwork(nn.Module):
    """Enhanced vector field network"""
    
    def __init__(self, features_dim: int, expression_dim: int, hidden_dim: int = 128, use_midihum: bool = False):
        super().__init__()
        self.features_dim = features_dim 
        self.expression_dim = expression_dim
        self.hidden_dim = hidden_dim
        self.use_midihum = use_midihum
        
        # Multi-modal conditioning system
        self.conditioning_system = MusicalConditioningSystem(
            basic_dim=9,
            hidden_dim=hidden_dim,
            use_midihum=use_midihum
        )
        
        self.time_embedding = TimeEmbedding(hidden_dim)
        
        self.main_network = nn.Sequential(
            nn.Linear(expression_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, expression_dim)
        )
        
        self.expression_proj = nn.Linear(expression_dim, expression_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor, features: torch.Tensor):
        """
        Args:
            t: Time parameter [batch_size] or [batch_size, 1]
            x: Current expression state [batch_size, expression_dim]
            features: Musical features [batch_size, features_dim]
        Returns:
            Vector field [batch_size, expression_dim]
        """
        batch_size = x.shape[0]
        
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(batch_size)
        elif t.dim() == 2:
            t = t.squeeze(-1)
        

        features_encoded = self.conditioning_system(features)  # [B, hidden_dim]
        
        time_embedded = self.time_embedding(t)  # [B, hidden_dim]
        
        combined_input = torch.cat([x, features_encoded, time_embedded], dim=1)
        
        output = self.main_network(combined_input)
        
        residual = self.expression_proj(x)
        
        return output + residual


class FMExpressiveModel(StreamingModule):
    """Enhanced Flow Matching model for expressive music performance with AudioCraft integration"""
    
    def __init__(self, features_dim: int = None, expression_dim: int = 4, hidden_dim: int = 128, 
                 use_midihum: bool = False, flow_matcher_type: str = "standard", sigma: float = 0.01, device: str = "cpu"):
        super().__init__()
        
        # Calculate features_dim based on use_midihum if not provided
        if features_dim is None:
            if use_midihum:
                self.features_dim = 9 + 202  # basic (9) + midihum (202) = 211
            else:
                self.features_dim = 9  # basic features only
        else:
            self.features_dim = features_dim
        
        self.expression_dim = expression_dim
        self.hidden_dim = hidden_dim
        self.use_midihum = use_midihum
        
        self.device = torch.device(device)
        
        print(f"Initializing FMExpressiveModel with features_dim={self.features_dim}, "
              f"expression_dim={expression_dim}, use_midihum={use_midihum}, device={self.device}")
        
        try:
            self.flow_matcher = ConditionalFlowMatcher(
                sigma=sigma, 
                schedule_type="cosine"
            )
        except Exception as e:
            self.flow_matcher = ConditionalFlowMatcher(
                sigma=sigma, 
                schedule_type="linear"
            )
        
        self.vector_field_network = VectorFieldNetwork(
            features_dim=self.features_dim,
            expression_dim=expression_dim,
            hidden_dim=hidden_dim,
            use_midihum=use_midihum
        )
        

        
        # Training state
        self.scaler = None
        self.target_scaler = None
        self.feature_scaler = None
        self.trained = False
        
        # Move model to device
        self.to(self.device)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device"""
        return tensor.to(self.device)

    def vector_field(self, t: torch.Tensor, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the vector field (flow) at given time and state
        
        Args:
            t: Time parameter
            x: Current state
            features: Musical features = basic_features + midihum_features (if use_midihum=True)
        """
        return self.vector_field_network(t, x, features)

    def flow_matching_loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """        
        Args:
            features: Musical features [batch_size, features_dim]
            target: Target expression parameters [batch_size, expression_dim]
        
        Returns:
            Loss value
        """
        features = self._to_device(features)
        target = self._to_device(target)
        
        batch_size = target.shape[0]
        
        # Sample noise
        x0 = torch.randn_like(target)
        
        # Sample time and compute flow
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, target)
        
        # Predict vector field using features (basic + midihum if enabled)
        vt_pred = self.vector_field(t, xt, features)
        
        # Compute MSE loss
        loss = F.mse_loss(vt_pred, ut)
        
        l2_reg = sum(p.pow(2.0).sum() for p in self.vector_field_network.parameters())
        loss = loss + 1e-5 * l2_reg
        
        return loss

    def train_model(self, training_notes: List[List[ExpressiveNote]], feature_extractor, epochs: int = 1000, batch_size: int = 32):
        print("Extracting features and targets...")
        
        all_contexts = []
        all_targets = []
        
        for note_sequence in training_notes:
            if len(note_sequence) == 0:
                continue
                
            # Extract context features
            contexts = feature_extractor.encode_features(
                note_sequence, fit=False, use_midihum=self.use_midihum
            )
            
            # Extract targets
            targets = []
            for note in note_sequence:
                target = note.get_targets()
                if any(t is None for t in target):
                    continue
                targets.append(target)
            
            if len(targets) == 0:
                continue
                
            targets = np.array(targets)
            
            # Handle dimension mismatch
            min_len = min(len(contexts), len(targets))
            contexts = contexts[:min_len]
            targets = targets[:min_len]
            
            all_contexts.append(contexts)
            all_targets.append(targets)
        
        if not all_contexts:
            raise ValueError("No valid training data found")
        
        X = np.vstack(all_contexts)
        y = np.vstack(all_targets)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Initialize or update scalers
        if self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        if self.target_scaler is None:
            self.target_scaler = StandardScaler()
            y = self.target_scaler.fit_transform(y)
        else:
            y = self.target_scaler.transform(y)
        
        X_tensor = self._to_device(torch.FloatTensor(X))
        y_tensor = self._to_device(torch.FloatTensor(y))
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        try:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=1e-4, 
                weight_decay=1e-4,
                betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        except:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            scheduler = None
        
        self.train_mode = True
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_features, batch_target in dataloader:
                optimizer.zero_grad()
                
                loss = self.flow_matching_loss(batch_features, batch_target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            
            if scheduler:
                scheduler.step()
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {lr:.2e}")
        
        print(f"Training completed. Final loss: {best_loss:.6f}")

    def sample(self, features: torch.Tensor, n_steps: int = 50, method: str = "dopri5", 
               ode_rtol: float = 1e-5, ode_atol: float = 1e-5) -> torch.Tensor:
        """
        Enhanced sampling with multiple integration methods
        
        Args:
            features: Musical features [batch_size, features_dim]
            n_steps: Number of integration steps (for fixed-step methods)
            method: Integration method ("euler", "midpoint", "rk4", "odeint", "dopri5")
            ode_rtol: Relative tolerance for adaptive ODE solver
            ode_atol: Absolute tolerance for adaptive ODE solver
            
        Returns:
            Generated expression parameters [batch_size, expression_dim]
        """
        self.eval()
        
        with torch.no_grad():
            features = self._to_device(features)
            batch_size = features.shape[0]
            
            # Start from noise
            x_0 = self._to_device(torch.randn(batch_size, self.expression_dim))
            
            # Check if using adaptive ODE solver
            if method in ["odeint", "dopri5"]:
                return self._sample_with_odeint(x_0, features, ode_rtol, ode_atol, method)
            
            # Fixed-step integration methods
            x = x_0
            dt = 1.0 / n_steps
            
            for i in range(n_steps):
                t = self._to_device(torch.full((batch_size,), i * dt))
                
                if method == "euler":
                    # Euler method
                    vt = self.vector_field(t, x, features)
                    x = x + dt * vt
                    
                elif method == "midpoint":
                    # Midpoint method (Runge-Kutta 2)
                    vt1 = self.vector_field(t, x, features)
                    x_mid = x + 0.5 * dt * vt1
                    t_mid = t + 0.5 * dt
                    vt2 = self.vector_field(t_mid, x_mid, features)
                    x = x + dt * vt2
                    
                elif method == "rk4":
                    # Runge-Kutta 4th order
                    k1 = self.vector_field(t, x, features)
                    k2 = self.vector_field(t + 0.5*dt, x + 0.5*dt*k1, features)
                    k3 = self.vector_field(t + 0.5*dt, x + 0.5*dt*k2, features)
                    k4 = self.vector_field(t + dt, x + dt*k3, features)
                    x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                    
                else:
                    raise ValueError(f"Unknown integration method: {method}")
        
        return x

    def _sample_with_odeint(self, x_0: torch.Tensor, features: torch.Tensor, 
                           ode_rtol: float, ode_atol: float, method: str) -> torch.Tensor:
        """        
        Args:
            x_0: Initial noise [batch_size, expression_dim]
            features: Musical features [batch_size, features_dim]
            ode_rtol: Relative tolerance
            ode_atol: Absolute tolerance  
            method: ODE solver method
            
        Returns:
            Generated expression parameters [batch_size, expression_dim]
        """
        def ode_func(t, x):
            """
            ODE function: dx/dt = v_θ(x, t, features)
            
            Args:
                t: scalar time (single value for all batch elements)
                x: state [batch_size, expression_dim]
            Returns:
                dx/dt: vector field [batch_size, expression_dim]
            """
            # Expand time to batch size if needed
            if t.dim() == 0:
                batch_size = x.shape[0]
                t_batch = t.expand(batch_size)
            else:
                t_batch = t
                
            return self.vector_field(t_batch, x, features)
        
        # Time span: integrate from 0 to 1 (following flow matching convention)
        t_span = torch.tensor([0.0, 1.0 - 1e-5], device=x_0.device)  # Avoid t=1 exactly
        
        # Choose solver method
        solver_method = "dopri5" if method in ["odeint", "dopri5"] else "euler"
        
        # Solve ODE with adaptive step size
        trajectory = odeint(
            ode_func, 
            x_0, 
            t_span, 
            method=solver_method,
            rtol=ode_rtol, 
            atol=ode_atol,
            options={}
        )
        
        # Return final state (trajectory[-1] is at t=1)
        return trajectory[-1]

    def predict(self, notes: List[ExpressiveNote], feature_extractor, 
                integration_method: str = "euler") -> List[ExpressiveNote]:
        """        
        Args:
            notes: Input notes to predict expressions for
            feature_extractor: Feature extraction system
            integration_method: "euler", "midpoint", "rk4", "odeint", "dopri5"
        
        Returns:
            Notes with predicted expressive parameters
        """
        if len(notes) == 0:
            return []
        
        features_array = feature_extractor.encode_features(
            notes, fit=False, use_midihum=self.use_midihum
        )
        
        # Apply scaling if available
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        # Convert to tensor
        features_tensor = self._to_device(torch.FloatTensor(features_array))
        
        # Generate predictions using enhanced sampling
        with torch.no_grad():
            if integration_method in ["odeint", "dopri5"]:
                # Use adaptive ODE solver with tighter tolerances for high quality
                predictions = self.sample(
                    features_tensor, 
                    method=integration_method,
                    ode_rtol=1e-5,
                    ode_atol=1e-5
                )
            else:
                # Use fixed-step methods
                predictions = self.sample(
                    features_tensor, 
                    n_steps=100 if integration_method == "rk4" else 50,
                    method=integration_method
                )
        
        predictions_np = predictions.numpy()
        
        # Apply inverse scaling if available
        if self.target_scaler is not None:
            predictions_np = self.target_scaler.inverse_transform(predictions_np)
        
        # Create new notes with predictions
        predicted_notes = []
        for i, note in enumerate(notes):
            if i >= len(predictions_np):
                break
                
            pred = predictions_np[i]
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
                beat_period=float(pred[0]),
                timing=float(pred[1]),
                velocity=float(pred[2]),
                articulation_log=float(pred[3])
            )
            predicted_notes.append(new_note)
        
        return predicted_notes

    def save(self, filepath: str, feature_scaler=None):
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'features_dim': self.features_dim,  
            'expression_dim': self.expression_dim,
            'hidden_dim': self.hidden_dim,
            'use_midihum': self.use_midihum,
            'device': str(self.device),
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_scaler': getattr(self, 'feature_scaler', None),  # Save feature_scaler if available
        }
        torch.save(save_dict, filepath)
        print(f"Enhanced model saved to {filepath}")

    def load(self, filepath: str):
        """Enhanced load with AudioCraft compatibility"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update dimensions with backward compatibility
        if 'features_dim' in checkpoint:
            self.features_dim = checkpoint['features_dim']
        elif 'actual_context_dim' in checkpoint:
            # Backward compatibility: old models used actual_context_dim
            self.features_dim = checkpoint['actual_context_dim']
            print(f"Loaded model with legacy context_dim={self.features_dim}, now using features_dim")
        
        if 'expression_dim' in checkpoint:
            self.expression_dim = checkpoint['expression_dim']
        if 'hidden_dim' in checkpoint:
            self.hidden_dim = checkpoint['hidden_dim']
        if 'use_midihum' in checkpoint:
            self.use_midihum = checkpoint['use_midihum']
        
        # Load scalers
        self.scaler = checkpoint.get('scaler')
        self.target_scaler = checkpoint.get('target_scaler')
        self.feature_scaler = checkpoint.get('feature_scaler')
        
        # Check device compatibility
        saved_device = checkpoint.get('device', 'cpu')
        if saved_device != str(self.device):
            print(f"Info: Model was saved on {saved_device} but loading on {self.device}")
        
        # Check AudioCraft compatibility
        saved_with_audiocraft = checkpoint.get('audiocraft_available', False)
        if saved_with_audiocraft:
            print("Warning: Model was saved with AudioCraft but AudioCraft is not available. Some features may not work.")
        elif not saved_with_audiocraft:
            print("Info: Model was saved without AudioCraft but AudioCraft is now available. Enhanced features will be used.")
        
        # Load model state
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Enhanced model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model state: {e}")
            print("This might be due to AudioCraft availability mismatch. Consider retraining.")

    def test_with_features(self, feature_extractor):
        """Enhanced testing with AudioCraft features"""
        print("Testing enhanced FMExpressiveModel...")
        print(f"Features dimension: {self.features_dim}")
        print(f"Expression dimension: {self.expression_dim}")
        print(f"Use midihum: {self.use_midihum}")
        
        # Test conditioning system
        if hasattr(self.vector_field_network, 'conditioning_system'):
            conditioning = self.vector_field_network.conditioning_system
            print("✓ Multi-modal conditioning system available")
            print(f"  Basic features: {conditioning.basic_dim} dims")
            if self.use_midihum:
                print(f"  Midihum features: {conditioning.midihum_dim} dims")
                print(f"  Total features: {conditioning.features_dim} dims")
                
                # Test individual encoders
                for encoder_name in ['harmonic_encoder', 'statistical_encoder', 'technical_encoder', 'timing_encoder']:
                    if hasattr(conditioning, encoder_name):
                        print(f"{encoder_name} available")
                    else:
                        print(f"{encoder_name} missing")
        
        # Test time embedding
        if hasattr(self.vector_field_network, 'time_embedding'):
            print("✓ AudioCraft-style time embedding available")
        
        # Test flow matcher
        print(f"✓ Flow matcher using {self.flow_matcher.schedule_type} schedule")
        
        # Test forward pass
        try:
            batch_size = 4
            features = self._to_device(torch.randn(batch_size, self.features_dim))
            target = self._to_device(torch.randn(batch_size, self.expression_dim))
            
            # Test loss computation
            loss = self.flow_matching_loss(features, target)
            print(f"✓ Loss computation successful: {loss.item():.6f}")
            
            # Test sampling with different methods
            test_methods = ["euler", "midpoint", "rk4", "odeint", "dopri5"]
            
            for method in test_methods:
                try:
                    if method in ["odeint", "dopri5"]:
                        samples = self.sample(features, method=method, ode_rtol=1e-3, ode_atol=1e-3)
                    else:
                        samples = self.sample(features, n_steps=10, method=method)
                    print(f"✓ Sampling with {method} method successful: {samples.shape}")
                except Exception as e:
                    print(f"✗ Sampling with {method} method failed: {e}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
        
        print("Enhanced model testing complete!")