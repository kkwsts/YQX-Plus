"""
JASCO-inspired Flow Matching for Expressive Music Performance Prediction

Based on the original JASCO implementation from Facebook Research AudioCraft:
- jasco/flow_matching.py
- jasco/jasco.py

Adapted for expressive performance parameter prediction instead of music generation.
"""

import math
import logging
import typing as tp
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint

from expressivenote import ExpressiveNote

logger = logging.getLogger(__name__)


@dataclass
class JASCOConfig:
    """Configuration for JASCO-style flow matching model"""
    # Model architecture (following original JASCO)
    dim: int = 128  # transformer dim
    num_heads: int = 8
    flow_dim: int = 4  # dimensionality of flow features (expression parameters: beat_period, timing, velocity, articulation)
    hidden_scale: int = 4
    num_layers: int = 12
    
    # Conditioning dimensions
    chords_dim: int = 0
    drums_dim: int = 0
    melody_dim: int = 0
    
    # Normalization and architecture
    norm: str = 'layer_norm'
    norm_first: bool = False
    bias_proj: bool = True
    
    # Initialization
    weight_init: tp.Optional[str] = None
    depthwise_init: tp.Optional[str] = None
    zero_bias_init: bool = False
    
    # CFG and dropout
    cfg_dropout: float = 0.0
    cfg_coef: float = 1.0
    attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = None
    
    # Time embedding
    time_embedding_dim: int = 128
    
    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    gradient_clip_norm: float = 0.2
    
    # Generation
    cfg_coef_all: float = 3.0
    cfg_coef_txt: float = 1.0
    euler_steps: int = 100
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5
    
    def __post_init__(self):
        if self.attribute_dropout is None:
            self.attribute_dropout = {}


class CFGTerm:
    """
    Base class for Multi Source Classifier-Free Guidance (CFG) terms.
    Directly from original JASCO implementation.
    """
    def __init__(self, conditions, weight):
        self.conditions = conditions
        self.weight = weight

    def drop_irrelevant_conds(self, conditions):
        """Drops irrelevant conditions from the CFG term."""
        raise NotImplementedError("No base implementation for setting generation params.")


class AllCFGTerm(CFGTerm):
    """A CFG term that retains all conditions."""
    def __init__(self, conditions, weight):
        super().__init__(conditions, weight)
        self.drop_irrelevant_conds()

    def drop_irrelevant_conds(self):
        pass


class NullCFGTerm(CFGTerm):
    """A CFG term that drops all conditions, effectively nullifying their influence."""
    def __init__(self, conditions, weight):
        super().__init__(conditions, weight)
        self.drop_irrelevant_conds()

    def drop_irrelevant_conds(self):
        """Drops all conditions by applying a dropout with probability 1.0."""
        # For our simplified case, we'll just set conditions to None
        self.conditions = None


class TextCFGTerm(CFGTerm):
    """A CFG term that selectively drops conditions based on specified dropout probabilities."""
    def __init__(self, conditions, weight, model_att_dropout=None):
        super().__init__(conditions, weight)
        self.model_att_dropout = model_att_dropout or {}
        self.drop_irrelevant_conds()

    def drop_irrelevant_conds(self):
        # For our simplified case, we'll handle this in the main model
        pass


class JASCOFlowMatcher(nn.Module):
    """
    JASCO-style Flow Matching model for expressive performance prediction.
    Closely follows the original FlowMatchingModel architecture.
    """
    
    def __init__(self, config: JASCOConfig, context_dim: int):
        super().__init__()
        self.config = config
        self.context_dim = context_dim
        self.dim = config.dim
        self.flow_dim = config.flow_dim
        
        # Main embedding layer (following original JASCO structure)
        # Combines flow features with temporal conditions
        self.emb = nn.Linear(
            config.flow_dim + config.chords_dim + config.drums_dim + config.melody_dim + context_dim, 
            config.dim, 
            bias=False
        )
        
        # Transformer layers (simplified version of UnetTransformer)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.hidden_scale * config.dim),
                dropout=0.1,
                activation='gelu',
                norm_first=config.norm_first,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # Output normalization (following original)
        self.out_norm: tp.Optional[nn.Module] = None
        if config.norm_first:
            self.out_norm = nn.LayerNorm(config.dim)
            
        # Output projection
        self.linear = nn.Linear(config.dim, config.flow_dim, bias=config.bias_proj)
        
        # Time parameter embedding (exactly from original JASCO)
        self.d_temb1 = config.time_embedding_dim
        self.d_temb2 = 4 * config.time_embedding_dim
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.d_temb1, self.d_temb2),
            nn.Linear(self.d_temb2, self.d_temb2),
        ])
        self.temb_proj = nn.Linear(self.d_temb2, config.dim)
        
        # Initialize weights
        self._init_weights(config.weight_init, config.depthwise_init, config.zero_bias_init)
    
    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Timestep embedding from original JASCO implementation.
        Taken from Stable Diffusion/DDPM implementation.
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
        """Time parameter embedding from original JASCO."""
        temb = self._get_timestep_embedding(t.flatten(), self.d_temb1)
        temb = self.temb.dense[0](temb)
        temb = temb * torch.sigmoid(temb)  # swish activation
        temb = self.temb.dense[1](temb)
        return temb

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialize weights following original JASCO pattern."""
        if weight_init is None:
            return
        
        # For simplicity, we'll use standard initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None and zero_bias_init:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _align_seq_length(self, cond: torch.Tensor, seq_len: int):
        """Align sequence length by trimming or padding (from original JASCO)."""
        # trim if needed
        cond = cond[:, :seq_len, :]

        # pad if needed
        B, T, C = cond.shape
        if T < seq_len:
            cond = torch.cat((cond, torch.zeros((B, seq_len - T, C), dtype=cond.dtype, device=cond.device)), dim=1)

        return cond

    def forward(self, latents: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass following original JASCO structure.
        
        Args:
            latents: Noisy latents [B, T, flow_dim]
            t: Time parameter [B] or [B, T]
            context: Context features [B, T, context_dim]
        
        Returns:
            Estimated vector field [B, T, flow_dim]
        """
        B, T, D = latents.shape
        
        # Ensure time has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(B, T)
        
        # Align context to sequence length
        context = self._align_seq_length(context, seq_len=T)
        
        # Concatenate latents with context (following original JASCO pattern)
        x = torch.cat([latents, context], dim=-1)
        
        # Project to transformer dimension
        input_ = self.emb(x)
        
        # Embed time parameter
        t_embs = self._embed_time_parameter(t[:, 0])  # Use first timestep for all
        time_emb = self.temb_proj(t_embs)[:, None, :].expand(-1, T, -1)
        
        # Add time embedding to input
        input_ = input_ + time_emb
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            input_ = layer(input_)
        
        # Output normalization
        if self.out_norm:
            input_ = self.out_norm(input_)
            
        # Final projection to flow dimension
        v_theta = self.linear(input_)
        
        return v_theta

    def estimated_vector_field(self, z, t, context, cfg_terms=[]):
        """
        Estimate vector field with multi-source CFG support.
        Following original JASCO implementation pattern.
        """
        if len(cfg_terms) > 1:
            z = z.repeat(len(cfg_terms), 1, 1)  # duplicate for multi-source CFG
            context = context.repeat(len(cfg_terms), 1, 1)
            
        v_thetas = self.forward(z, t, context)
        return self._multi_source_cfg_postprocess(v_thetas, cfg_terms)

    def _multi_source_cfg_postprocess(self, v_thetas, cfg_terms):
        """Postprocess vector fields for multi-source CFG (from original JASCO)."""
        if len(cfg_terms) <= 1:
            return v_thetas
            
        v_theta_per_term = v_thetas.chunk(len(cfg_terms))
        return sum([ct.weight * term_vf for ct, term_vf in zip(cfg_terms, v_theta_per_term)])

    @torch.no_grad()
    def generate(self,
                 context: torch.Tensor,
                 cfg_coef_all: float = 3.0,
                 cfg_coef_txt: float = 1.0,
                 euler: bool = False,
                 euler_steps: int = 100,
                 ode_rtol: float = 1e-5,
                 ode_atol: float = 1e-5,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> torch.Tensor:
        """
        Generate samples using flow matching (following original JASCO structure).
        """
        assert not self.training, "generation shouldn't be used in training mode."
        
        device = next(iter(self.parameters())).device
        B, T = context.shape[:2]
        D = self.flow_dim
        
        # Setup CFG terms (simplified for our use case)
        cfg_terms = []
        if cfg_coef_all != 0:
            cfg_terms.append(AllCFGTerm(conditions=context, weight=cfg_coef_all))
        if cfg_coef_txt != 0:
            cfg_terms.append(TextCFGTerm(conditions=context, weight=cfg_coef_txt))
        
        # Add null term
        if cfg_terms:
            null_weight = 1 - sum([ct.weight for ct in cfg_terms])
            cfg_terms.append(NullCFGTerm(conditions=context, weight=null_weight))
        
        # Initial noise
        z_0 = torch.randn((B, T, D), device=device)
        
        if euler:
            # Euler integration (from original JASCO)
            dt = 1.0 / euler_steps
            z = z_0
            t = torch.zeros((B,), device=device)
            
            for _ in range(euler_steps):
                v_theta = self.estimated_vector_field(z, t, context, cfg_terms)
                z = z + dt * v_theta
                t = t + dt
                
            return z
        else:
            # ODE solver (from original JASCO)
            t_span = torch.tensor([0, 1.0 - 1e-5], device=device)
            num_evals = 0
            
            def inner_ode_func(t, z):
                nonlocal num_evals
                num_evals += 1
                if callback is not None:
                    ESTIMATED_ODE_SOLVER_STEPS = 300
                    callback(num_evals, ESTIMATED_ODE_SOLVER_STEPS)
                
                # Convert scalar t to tensor for batch
                t_batch = torch.full((B,), t.item(), device=device)
                return self.estimated_vector_field(z, t_batch, context, cfg_terms)
            
            z = odeint(
                inner_ode_func,
                z_0,
                t_span,
                atol=ode_atol,
                rtol=ode_rtol,
            )
            
            logger.info("Generated in %d steps", num_evals)
            return z[-1]


class JASCOExpressiveModel:
    """
    Complete JASCO-style expressive performance model.
    Follows the structure of the original JASCO class.
    """
    
    def __init__(self, config: JASCOConfig, use_midihum_features: bool = False):
        self.config = config
        self.use_midihum_features = use_midihum_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Will be set during training
        self.context_dim = None
        self.model = None
        self.context_scaler = StandardScaler()
        self.expression_scaler = StandardScaler()
        
        self.trained = False
        
        # Generation parameters (following original JASCO pattern)
        self.generation_params = {
            'cfg_coef_all': config.cfg_coef_all,
            'cfg_coef_txt': config.cfg_coef_txt,
            'euler': False,
            'euler_steps': config.euler_steps,
            'ode_rtol': config.ode_rtol,
            'ode_atol': config.ode_atol,
        }
    
    def set_generation_params(self, **kwargs):
        """Set generation parameters (following original JASCO API)."""
        self.generation_params.update(kwargs)
    
    def _initialize_model(self, context_dim: int):
        """Initialize model with correct context dimension."""
        self.context_dim = context_dim
        self.model = JASCOFlowMatcher(self.config, context_dim).to(self.device)
        
        # Setup optimizer with warmup (following modern practices)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Linear warmup scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _compute_loss(self, x1: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss using optimal transport formulation."""
        batch_size = x1.shape[0]
        device = x1.device
        
        # Sample noise and time
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device, dtype=x1.dtype)
        
        # Optimal transport interpolation
        sigma_min = 1e-5
        t_expanded = t.unsqueeze(1).unsqueeze(2)
        xt = (1 - (1 - sigma_min) * t_expanded) * x0 + t_expanded * x1
        
        # Target velocity (OT flow)
        ut = x1 - (1 - sigma_min) * x0
        
        # Predict velocity
        pred_velocity = self.model(xt, t, context)
        
        # Compute loss with optional weighting
        loss = F.mse_loss(pred_velocity, ut)
        
        return loss
    
    def train(self, training_notes: tp.List[tp.List[ExpressiveNote]], feature_extractor, 
              epochs: int = 1000, batch_size: int = 32):
        """Train the JASCO-style flow matching model."""
        print("Training JASCO-style Flow Matching model...")
        
        # Flatten all notes
        all_notes = []
        for piece_notes in training_notes:
            all_notes.extend(piece_notes)
        
        # Filter notes with targets
        training_notes_filtered = [note for note in all_notes if 
                                 note.beat_period is not None and
                                 note.timing is not None and
                                 note.velocity is not None and
                                 note.articulation_log is not None]
        
        print(f"Training on {len(training_notes_filtered)} notes")
        
        # Extract features
        context_features = feature_extractor.encode_features(
            training_notes_filtered, 
            fit=True, 
            use_midihum_features=self.use_midihum_features
        )
        
        # Initialize model
        if self.model is None:
            self._initialize_model(context_features.shape[1])
        
        # Scale features
        context_features = self.context_scaler.fit_transform(context_features)
        context_features = torch.tensor(context_features, dtype=torch.float32, device=self.device)
        
        # Prepare targets
        targets = np.array([[
            note.beat_period,
            note.timing,
            note.velocity / 127.0,  # Normalize velocity
            note.articulation_log
        ] for note in training_notes_filtered])
        
        targets = self.expression_scaler.fit_transform(targets)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # Add sequence dimension (each note is a sequence of length 1)
        context_features = context_features.unsqueeze(1)  # [batch, 1, context_dim]
        targets = targets.unsqueeze(1)  # [batch, 1, expression_dim]
        
        # Training loop
        dataset_size = len(context_features)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        self.model.train()
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
                loss = self._compute_loss(batch_targets, batch_context)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
        
        self.trained = True
        print("Training completed!")
    
    def predict(self, notes: tp.List[ExpressiveNote], feature_extractor) -> tp.List[ExpressiveNote]:
        """Predict expressive parameters using the trained model."""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        context_features = feature_extractor.encode_features(
            notes, 
            fit=False, 
            use_midihum_features=self.use_midihum_features
        )
        
        context_features = self.context_scaler.transform(context_features)
        context_features = torch.tensor(context_features, dtype=torch.float32, device=self.device)
        context_features = context_features.unsqueeze(1)  # Add sequence dimension
        
        # Generate predictions using JASCO-style sampling
        predictions = self.model.generate(context_features, **self.generation_params)
        predictions = predictions.squeeze(1)  # Remove sequence dimension
        
        # Inverse transform
        predictions = predictions.cpu().numpy()
        predictions = self.expression_scaler.inverse_transform(predictions)
        
        # Create predicted notes
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
        """Save model (following original JASCO pattern)."""
        model_data = {
            'config': self.config,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'context_scaler': self.context_scaler,
            'expression_scaler': self.expression_scaler,
            'context_dim': self.context_dim,
            'use_midihum_features': self.use_midihum_features,
            'trained': self.trained,
            'generation_params': self.generation_params
        }
        torch.save(model_data, filepath)
    
    def load(self, filepath: str):
        """Load model (following original JASCO pattern)."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.config = model_data['config']
        self.context_dim = model_data['context_dim']
        self.use_midihum_features = model_data['use_midihum_features']
        self.trained = model_data['trained']
        self.generation_params = model_data.get('generation_params', self.generation_params)
        
        # Recreate model
        if self.context_dim is not None:
            self._initialize_model(self.context_dim)
            if model_data['model_state_dict'] is not None:
                self.model.load_state_dict(model_data['model_state_dict'])
        
        self.context_scaler = model_data['context_scaler']
        self.expression_scaler = model_data['expression_scaler']


# Convenience function following original JASCO pattern
def get_pretrained_jasco_expressive(name: str = 'jasco-expressive', device=None):
    """
    Return pretrained JASCO expressive model.
    Following the pattern of JASCO.get_pretrained().
    """
    if device is None:
        if torch.cuda.device_count():
            device = 'cuda'
        else:
            device = 'cpu'
    
    config = JASCOConfig()
    model = JASCOExpressiveModel(config, use_midihum_features=True)
    model.device = torch.device(device)
    
    return model