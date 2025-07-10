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
from audiocraft.modules.transformer import StreamingTransformerLayer, create_norm_fn
from audiocraft.modules.unet_transformer import UnetTransformer

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher



class AudioCraftTransformerBlock(nn.Module):
    """Transformer block for musical context processing"""
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
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
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


class VectorFieldNetwork(nn.Module):
    
    def __init__(self, features_dim: int, expression_dim: int, hidden_dim: int = 128, 
                 use_midihum: bool = False, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.features_dim = features_dim 
        self.expression_dim = expression_dim
        self.hidden_dim = hidden_dim
        self.use_midihum = use_midihum
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if use_midihum:
            input_features_dim = 6 + 171  
        else:
            input_features_dim = 9 
            
        self.time_embedding = TimeEmbedding(hidden_dim)

        self.input_embedding = nn.Linear(expression_dim + input_features_dim, hidden_dim, bias=False)
        
        try:
            self.transformer = UnetTransformer(
                d_model=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=int(4 * hidden_dim),
                norm='layer_norm',
                norm_first=True,
                layer_class=StreamingTransformerLayer,
                num_layers=num_layers,
                dropout=0.1,
                activation='gelu'
            )
            self.uses_audiocraft = True
        except Exception as e:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=int(4 * hidden_dim),
                    dropout=0.1,
                    activation='gelu',
                    norm_first=True,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            self.uses_audiocraft = False
            
        self.out_norm = create_norm_fn('layer_norm', hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, expression_dim, bias=True)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

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
        
        # Time embedding
        time_embedded = self.time_embedding(t)  # [B, hidden_dim]
        
        # Combine expression state and features
        combined_input = torch.cat([x, features], dim=1)  # [B, expression_dim + features_dim]
        
        # Input embedding
        embedded = self.input_embedding(combined_input)  # [B, hidden_dim]
        
        # Add time embedding
        embedded = embedded + time_embedded  # [B, hidden_dim]
        
        # Add sequence dimension for transformer
        embedded = embedded.unsqueeze(1)  # [B, 1, hidden_dim]
        
        if self.uses_audiocraft:
            # AudioCraft UnetTransformer
            transformer_output = self.transformer(embedded)  # [B, 1, hidden_dim]
        else:
            # PyTorch Transformer
            transformer_output = self.transformer(embedded)  # [B, 1, hidden_dim]
            
        # Remove sequence dimension
        transformer_output = transformer_output.squeeze(1)  # [B, hidden_dim]
        
        # Output normalization and projection
        normalized = self.out_norm(transformer_output)
        vector_field = self.output_projection(normalized)  # [B, expression_dim]
        
        return vector_field



class FMExpressiveModel(StreamingModule):
    
    def __init__(self, features_dim: int = None, expression_dim: int = 4, hidden_dim: int = 128, epochs: int = 1000,
                 use_midihum: bool = False, flow_matcher_type: str = "standard", sigma: float = 0.01, 
                 device: str = "cpu", num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        
        self.features_dim = features_dim
        
        self.expression_dim = expression_dim
        self.hidden_dim = hidden_dim
        self.use_midihum = use_midihum
        self.epochs = epochs
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.flow_matcher_type = flow_matcher_type
        
        self.device = torch.device(device)
        
        print(f"Initializing FMExpressiveModel with features_dim={self.features_dim}, "
              f"expression_dim={expression_dim}, use_midihum={use_midihum}, device={self.device}")
        
        if flow_matcher_type == "optimal_transport":
            self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        # elif flow_matcher_type == "target":
        #     self.flow_matcher = TargetConditionalFlowMatcher(sigma=sigma)
        else:
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        
        self.vector_field_network = VectorFieldNetwork(
            features_dim=self.features_dim,
            expression_dim=expression_dim,
            hidden_dim=hidden_dim,
            use_midihum=use_midihum,
            num_heads=num_heads,
            num_layers=num_layers
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
        # features = self._to_device(features)
        # target = self._to_device(target)
        batch_size = target.shape[0]
        
        x0 = torch.randn_like(target)
        x1 = target
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=target.device)
        
        epsilon = torch.randn_like(target)
        
        xt = self.flow_matcher.sample_xt(x0, x1, t, epsilon)
        ut = self.flow_matcher.compute_conditional_flow(x0, x1, t, xt)
        
        vt_pred = self.vector_field(t, xt, features)
        
        # Compute MSE loss
        loss = F.mse_loss(vt_pred, ut)

        
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
                note_sequence, fit=True, use_midihum=self.use_midihum
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
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            # num_workers=4, 
            # pin_memory=False, 
            # drop_last=True
        )
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4,
            betas=(0.9, 0.95), 
            # eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        
        self.train_mode = True
        best_loss = float('inf')
        patience = max(10, epochs // 10) 
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
            
            if (epoch + 1) % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {lr:.2e}")
        
        print(f"Training completed. Final loss: {best_loss:.6f}")

    def sample(self, features: torch.Tensor, n_steps: int = 50, method: str = "dopri5", 
               ode_rtol: float = 1e-5, ode_atol: float = 1e-5) -> torch.Tensor:
        """        
        Args:
            features: Musical features [batch_size, features_dim]
            n_steps: Number of integration steps (for fixed-step methods)
            method: Integration method ("euler", "odeint", "dopri5")
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
                t = torch.full((batch_size,), i * dt, device=x.device, dtype=x.dtype)
                
                if method == "euler":
                    # Euler method
                    vt = self.vector_field(t, x, features)
                    x = x + dt * vt
                    
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
        t_span = torch.tensor([0.0, 1.0 - 1e-5], device=x_0.device, dtype=x_0.dtype)  # Avoid t=1 exactly
        
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
    

    def predict(self, notes: List[ExpressiveNote], 
                integration_method: str = "dopri5") -> List[ExpressiveNote]:
        
        features_array = self.scaler.transform(notes)
        features_tensor = torch.tensor(features_array, dtype=torch.float32, device=self.device)
        
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
                predictions = self.sample(
                    features_tensor, 
                    n_steps=100 if integration_method == "euler" else 50,
                    method=integration_method
                )
        
    
        predictions = predictions.numpy()
        predictions = self.target_scaler.inverse_transform(predictions)
        
        return predictions

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
        print(f"Model saved to {filepath}")

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
        except Exception as e:
            print(f"Error loading model state: {e}")
            print("This might be due to AudioCraft availability mismatch. Consider retraining.")
