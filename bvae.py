import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from torchinfo import summary
import wandb
import math
from audiocraft.modules.transformer import StreamingTransformerLayer, create_norm_fn
from audiocraft.modules.unet_transformer import UnetTransformer
from expressivenote import ExpressiveNote

from torch.cuda.amp import autocast, GradScaler

class TransformerEncoder(nn.Module):
    """Transformer-based encoder for context features"""
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
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
        except:
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
        
        # Output normalization
        try:
            self.out_norm = create_norm_fn('layer_norm', hidden_dim)
        except:
            self.out_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, input_dim]
        Returns:
            Encoded features [batch_size, hidden_dim]
        """
        # Project to hidden dimension
        embedded = self.input_projection(x)  # [B, hidden_dim]
        
        # Add sequence dimension for transformer
        embedded = embedded.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Apply transformer
        transformer_output = self.transformer(embedded)  # [B, 1, hidden_dim]
        
        # Remove sequence dimension and normalize
        output = transformer_output.squeeze(1)  # [B, hidden_dim]
        output = self.out_norm(output)
        
        return output


class TransformerDecoder(nn.Module):
    """Transformer-based decoder for generating targets from latent + context"""
    def __init__(self, latent_dim: int, context_dim: int, target_dim: int, 
                 hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection (latent + context)
        input_dim = latent_dim + context_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
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
        except:
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
        
        # Output normalization and projection
        try:
            self.out_norm = create_norm_fn('layer_norm', hidden_dim)
        except:
            self.out_norm = nn.LayerNorm(hidden_dim)
        
        self.output_projection = nn.Linear(hidden_dim, target_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent representation [batch_size, latent_dim]
            context: Context features [batch_size, context_dim]
        Returns:
            Decoded targets [batch_size, target_dim]
        """
        combined_input = torch.cat([latent, context], dim=1)  # [B, latent_dim + context_dim]
        
        embedded = self.input_projection(combined_input)  # [B, hidden_dim]
        
        embedded = embedded.unsqueeze(1)  # [B, 1, hidden_dim]
        
        transformer_output = self.transformer(embedded)  # [B, 1, hidden_dim]
        
        output = transformer_output.squeeze(1)  # [B, hidden_dim]
        output = self.out_norm(output)
        output = self.output_projection(output)  # [B, target_dim]
        
        return output


class BetaVAE(nn.Module):
    """
    β-VAE for Expressive Performance Modeling
    
    Conditional β-VAE that maps musical context features to expressive performance parameters
    with controllable disentanglement via β parameter.
    
    Based on: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    Adapted for musical performance from PyTorch-VAE repository structure.
    """
    
    def __init__(self, 
                 context_dim: int = 9,
                 target_dim: int = 4, 
                 latent_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 beta: float = 1.0,
                 gamma: float = 10.0,
                 max_capacity: int = 10,
                 capacity_max_iter: int = 20,
                 use_midihum: bool = False,
                 device: Optional[torch.device] = None):
        super(BetaVAE, self).__init__()
        
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.capacity_max_iter = capacity_max_iter
        self.use_midihum = use_midihum
        self.num_iter = 0
        
        self.device = device
        
        # Build encoder: context -> latent (no target input to avoid leakage)
        self.encoder = TransformerEncoder(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Build decoder: latent + context -> target
        self.decoder = TransformerDecoder(
            latent_dim=latent_dim,
            context_dim=context_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Scalers for normalization
        self.context_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_scaler = None
        
        # Training state
        self.trained = False
        
        # Move to device
        self.to(self.device)
    
    def encode(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode context to latent distribution parameters"""
        h = self.encoder(context)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Decode latent + context to target parameters"""
        return self.decoder(z, context)
    
    def forward(self, context: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through β-VAE"""
        mu, log_var = self.encode(context)
        z = self.reparameterize(mu, log_var)
        
        recon_target = self.decode(z, context)
        
        return {
            'recon_x': recon_target,
            'target': target,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def loss_function(self, results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        β-VAE loss with capacity annealing
        
        Following Burgess et al. (2018) formulation
        """
        recons = results['recon_x']
        target = results['target']
        mu = results['mu']
        log_var = results['log_var']
        
        self.num_iter += 1
        
        # Reconstruction loss (MSE for continuous targets)
        recons_loss = F.mse_loss(recons, target, reduction='mean')
        
        # KL divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        
        # Capacity annealing (following original β-VAE paper)
        if self.gamma > 0:
            C = min(self.max_capacity, self.max_capacity * self.num_iter / self.capacity_max_iter)
            capacity_loss = self.gamma * torch.abs(kld_loss - C)
        else:
            capacity_loss = self.beta * kld_loss
        
        total_loss = recons_loss + capacity_loss
        
        return {
            'loss': total_loss,
            'Reconstruction_Loss': recons_loss,
            'KLD': kld_loss,
            'Capacity_Loss': capacity_loss
        }
    
    def sample(self, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate samples from prior"""
        batch_size = context.shape[0]
        
        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        samples = []
        for _ in range(num_samples):
            z_sample = torch.randn(batch_size, self.latent_dim).to(self.device)
            sample = self.decode(z_sample, context)
            samples.append(sample)
        
        if num_samples == 1:
            return samples[0]
        else:
            return torch.stack(samples, dim=1)
    
    def interpolate(self, context1: torch.Tensor, context2: torch.Tensor,
                   num_steps: int = 10) -> torch.Tensor:
        """Interpolate between two contexts in latent space"""
        with torch.no_grad():
            # Encode both contexts
            mu1, _ = self.encode(context1)
            mu2, _ = self.encode(context2)
            
            # Interpolate in latent space
            interpolations = []
            for alpha in torch.linspace(0, 1, num_steps):
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                context_interp = (1 - alpha) * context1 + alpha * context2
                
                recon = self.decode(z_interp, context_interp)
                interpolations.append(recon)
            
            return torch.stack(interpolations, dim=1)


class BVAEExpressiveModel:
    """
    Complete β-VAE expressive performance model
    Compatible with YQX system interface
    """
    
    def __init__(self, 
                 context_dim: int = 9,
                 target_dim: int = 4,
                 latent_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 beta: float = 4.0,
                 gamma: float = 1000.0,
                 use_midihum: bool = False,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.beta = beta
        self.gamma = gamma
        self.use_midihum = use_midihum
        self.learning_rate = learning_rate
        
        self.device = device
        
        # Model will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Scalers
        self.context_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_scaler = None
        
        self.trained = False
        
        self._initialize_model(context_dim)
    
    def _initialize_model(self, context_dim: int):
        """Initialize model with correct dimensions"""
        self.context_dim = context_dim
        self.model = BetaVAE(
            context_dim=context_dim,
            target_dim=self.target_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            beta=self.beta,
            gamma=self.gamma,
            use_midihum=self.use_midihum,
            device=self.device
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        
        try:
            self.scaler = GradScaler()
        except:
            self.scaler = None
        print(summary(self.model))
    
    def train(self, context_features: np.ndarray, targets: np.ndarray,
              val_features: np.ndarray = None, val_targets: np.ndarray = None,
              epochs: int = 1000, batch_size: int = 32):
        """Train the β-VAE model on pre-extracted features and targets"""
        print("Training β-VAE model...")
        print(f"Training on {len(context_features)} samples")
        
        # Scale features and targets
        context_features_scaled = self.context_scaler.fit_transform(context_features)
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        val_context_features_scaled = None
        val_targets_scaled = None
        if val_features is not None and val_targets is not None:
            print(f"Using validation set with {len(val_features)} samples")
            val_context_features_scaled = self.context_scaler.transform(val_features)
            val_targets_scaled = self.target_scaler.transform(val_targets)
        
        # Training loop
        dataset_size = len(context_features_scaled)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        self.model.train()
        best_loss = float('inf')
        best_epoch = 0
        patience = max(10, epochs // 10)
        patience_counter = 0
        
        best_model_state = None
        
        context_tensor = torch.from_numpy(context_features_scaled).float()
        targets_tensor = torch.from_numpy(targets_scaled).float()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kld_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(dataset_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                
                # Get batch indices
                batch_indices = perm[start_idx:end_idx]
                
                batch_context = context_tensor[batch_indices].to(self.device, non_blocking=True)
                batch_targets = targets_tensor[batch_indices].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True) 
                
                try:
                    with autocast():
                        results = self.model(batch_context, batch_targets)
                        loss_dict = self.model.loss_function(results)
                        loss = loss_dict['loss']
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except:
                    results = self.model(batch_context, batch_targets)
                    loss_dict = self.model.loss_function(results)
                    loss = loss_dict['loss']
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += loss_dict['Reconstruction_Loss'].item()
                epoch_kld_loss += loss_dict['KLD'].item()
                
                if batch_idx % 200 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.scheduler.step()
            
            # Validation phase
            val_loss = None
            val_recon_loss = None
            val_kld_loss = None
            if val_context_features_scaled is not None and val_targets_scaled is not None:
                self.model.eval()
                val_batch_size = min(batch_size, len(val_context_features_scaled))
                val_num_batches = (len(val_context_features_scaled) + val_batch_size - 1) // val_batch_size
                
                total_val_loss = 0.0
                total_val_recon_loss = 0.0
                total_val_kld_loss = 0.0
                
                with torch.no_grad():
                    for val_batch_idx in range(val_num_batches):
                        val_start_idx = val_batch_idx * val_batch_size
                        val_end_idx = min(val_start_idx + val_batch_size, len(val_context_features_scaled))
                        
                        val_batch_context = torch.from_numpy(
                            val_context_features_scaled[val_start_idx:val_end_idx]
                        ).float().to(self.device, non_blocking=True)
                        val_batch_targets = torch.from_numpy(
                            val_targets_scaled[val_start_idx:val_end_idx]
                        ).float().to(self.device, non_blocking=True)
                        
                        val_results = self.model(val_batch_context, val_batch_targets)
                        val_loss_dict = self.model.loss_function(val_results)
                        
                        total_val_loss += val_loss_dict['loss'].item()
                        total_val_recon_loss += val_loss_dict['Reconstruction_Loss'].item()
                        total_val_kld_loss += val_loss_dict['KLD'].item()
                    
                    # Average validation losses
                    val_loss = total_val_loss / val_num_batches
                    val_recon_loss = total_val_recon_loss / val_num_batches
                    val_kld_loss = total_val_kld_loss / val_num_batches
                
                self.model.train()
            
            current_loss = val_loss if val_loss is not None else (epoch_loss / num_batches)
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                best_model_state = self.model.state_dict().copy()
            
            # Log to wandb
            if val_loss is not None:
                wandb.log({
                    "epoch": epoch,
                    "bvae_total_loss": epoch_loss / num_batches,
                    "bvae_reconstruction_loss": epoch_recon_loss / num_batches,
                    "bvae_kld_loss": epoch_kld_loss / num_batches,
                    "bvae_capacity_loss": (epoch_loss - epoch_recon_loss - epoch_kld_loss) / num_batches,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "val_total_loss": val_loss,
                    "val_reconstruction_loss": val_recon_loss,
                    "val_kld_loss": val_kld_loss
                })
            else:
                wandb.log({
                    "epoch": epoch,
                    "bvae_total_loss": epoch_loss / num_batches,
                    "bvae_reconstruction_loss": epoch_recon_loss / num_batches,
                    "bvae_kld_loss": epoch_kld_loss / num_batches,
                    "bvae_capacity_loss": (epoch_loss - epoch_recon_loss - epoch_kld_loss) / num_batches,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
            
            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                avg_recon = epoch_recon_loss / num_batches
                avg_kld = epoch_kld_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                val_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}{val_str}, "
                      f"Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}, LR: {lr:.6f}")
        
        self.trained = True
        print("β-VAE training completed!")
        print(f"Best model was at epoch {best_epoch + 1} with loss: {best_loss:.6f}")
        
        self.best_model_state = best_model_state
        self.best_epoch = best_epoch
        self.best_loss = best_loss
    
    def predict(self, context_features: np.ndarray, num_samples: int = 1, batch_size: int = 32) -> np.ndarray:
        """Predict targets using β-VAE on pre-extracted features"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Scale features
        context_features_scaled = self.context_scaler.transform(context_features)
        
        context_tensor = torch.from_numpy(context_features_scaled).float()
        
        # Generate predictions
        num_batches = (len(context_features_scaled) + batch_size - 1) // batch_size
        
        all_predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(context_features_scaled))
                
                batch_context = context_tensor[start_idx:end_idx].to(self.device, non_blocking=True)
                
                batch_predictions = self.model.sample(batch_context, num_samples=num_samples)
                
                if num_samples > 1:
                    batch_predictions = batch_predictions.mean(dim=1)
                
                batch_predictions = batch_predictions.cpu().numpy()
                all_predictions.append(batch_predictions)
                
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        predictions = np.vstack(all_predictions)
        
        # Inverse transform
        predictions = self.target_scaler.inverse_transform(predictions)
        
        return predictions
    
    def get_latent_representation(self, context_features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get latent representations for analysis"""
        if not self.trained:
            raise ValueError("Model must be trained before encoding")
        
        context_features_scaled = self.context_scaler.transform(context_features)
        
        context_tensor = torch.from_numpy(context_features_scaled).float()
        
        num_batches = (len(context_features_scaled) + batch_size - 1) // batch_size
        
        all_latent_repr = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(context_features_scaled))
                
                batch_context = context_tensor[start_idx:end_idx].to(self.device, non_blocking=True)
                
                mu, _ = self.model.encode(batch_context)
                batch_latent = mu.cpu().numpy()
                all_latent_repr.append(batch_latent)
                
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        latent_repr = np.vstack(all_latent_repr)
        
        return latent_repr
    
    def save(self, filepath: str, feature_scaler=None, save_best: bool = True):
        """Save trained β-VAE model"""
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
            
        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'context_scaler': self.context_scaler,
            'target_scaler': self.target_scaler,
            'context_dim': self.context_dim,
            'target_dim': self.target_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'beta': self.beta,
            'gamma': self.gamma,
            'use_midihum': self.use_midihum,
            'learning_rate': self.learning_rate,
            'trained': self.trained,
            'feature_scaler': self.feature_scaler,
            'model_type': 'last'
        }
        torch.save(model_data, filepath)
        print(f"Last model saved to {filepath}")
        
        if save_best and hasattr(self, 'best_model_state') and self.best_model_state is not None:
            base_path = filepath.rsplit('.', 1)[0] 
            extension = filepath.rsplit('.', 1)[1] if '.' in filepath else 'pth'
            best_filepath = f"{base_path}_best.{extension}"
            
            best_model_data = {
                'model_state_dict': self.best_model_state,
                'context_scaler': self.context_scaler,
                'target_scaler': self.target_scaler,
                'context_dim': self.context_dim,
                'target_dim': self.target_dim,
                'latent_dim': self.latent_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'beta': self.beta,
                'gamma': self.gamma,
                'use_midihum': self.use_midihum,
                'learning_rate': self.learning_rate,
                'trained': self.trained,
                'feature_scaler': self.feature_scaler,
                'model_type': 'best',  # Indicate this is the best model
                'best_epoch': getattr(self, 'best_epoch', 0),
                'best_loss': getattr(self, 'best_loss', float('inf')),
            }
            torch.save(best_model_data, best_filepath)
            print(f"Best model saved to {best_filepath}")
            print(f"Best model was at epoch {getattr(self, 'best_epoch', 0) + 1} with loss: {getattr(self, 'best_loss', float('inf')):.6f}")
    
    def load(self, filepath: str):
        """Load trained β-VAE model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Restore parameters
        self.context_dim = model_data['context_dim']
        self.target_dim = model_data['target_dim']
        self.latent_dim = model_data['latent_dim']
        self.hidden_dim = model_data.get('hidden_dim', 128) 
        self.num_heads = model_data.get('num_heads', 4) 
        self.num_layers = model_data.get('num_layers', 2)
        self.beta = model_data['beta']
        self.gamma = model_data['gamma']
        self.use_midihum = model_data['use_midihum']
        self.learning_rate = model_data['learning_rate']
        self.trained = model_data['trained']
        self.feature_scaler = model_data.get('feature_scaler', None)
        
        # Recreate model
        if self.context_dim is not None:
            self._initialize_model(self.context_dim)
            if model_data['model_state_dict'] is not None:
                self.model.load_state_dict(model_data['model_state_dict'])
        
        # Restore scalers
        self.context_scaler = model_data['context_scaler']
        self.target_scaler = model_data['target_scaler']
        
        model_type = model_data.get('model_type', 'unknown')
        if model_type == 'best':
            best_epoch = model_data.get('best_epoch', 0)
            best_loss = model_data.get('best_loss', float('inf'))
            print(f"Loaded best model from epoch {best_epoch + 1} with loss: {best_loss:.6f}")
        elif model_type == 'last':
            print("Loaded last model (final epoch)")
        else:
            print("Loaded model (legacy format)")

    def model_summary(self):
        """Print detailed model architecture and parameter summary"""
        if self.model is None:
            print("Model not initialized yet. Please train or load a model first.")
            return
        
        if summary is None:
            print("Model summary not available. Please install torchinfo or torchsummary:")
            print("pip install torchinfo")
            return
        
        print("=" * 80)
        print("β-VAE Expressive Model Summary")
        print("=" * 80)
        
        # Wrapper configuration
        print("Wrapper Configuration:")
        print(f"  • Context dimension: {self.context_dim}")
        print(f"  • Target dimension: {self.target_dim}")
        print(f"  • Latent dimension: {self.latent_dim}")
        print(f"  • Hidden dimension: {self.hidden_dim}")
        print(f"  • Number of heads: {self.num_heads}")
        print(f"  • Number of layers: {self.num_layers}")
        print(f"  • Beta (β): {self.beta}")
        print(f"  • Gamma (γ): {self.gamma}")
        print(f"  • Use MidiHum features: {self.use_midihum}")
        print(f"  • Learning rate: {self.learning_rate}")
        print(f"  • Device: {self.device}")
        print(f"  • Trained: {self.trained}")
        print()
        
        # Scalers info
        print("Data Preprocessing:")
        print(f"  • Context scaler: {type(self.context_scaler).__name__}")
        print(f"  • Target scaler: {type(self.target_scaler).__name__}")
        print(f"  • Feature scaler: {type(self.feature_scaler).__name__ if self.feature_scaler else 'None'}")
        print()
        
        # Call the underlying model's summary using API
        print("Underlying β-VAE Model Architecture:")
        print("-" * 40)
        self.model.model_summary()
        
        # Training info
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            print("Training Setup:")
            print(f"  • Optimizer: {type(self.optimizer).__name__}")
            print(f"  • Learning rate: {self.learning_rate}")
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                print(f"  • Scheduler: {type(self.scheduler).__name__}")
            print()
        
        print("=" * 80) 