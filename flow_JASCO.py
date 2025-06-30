import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import math
from typing import List, Dict, Any, Optional
from expressivenote import *
from conditioners import *


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )    
    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.encoder(conditions)
    
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1))
    return emb

class ConditionalFlowMatchingModel(nn.Module):    
    def __init__(self, 
                 categorical_dim: int = 64,
                 continuous_dim: int = 64,
                 midihum_dim: int = 64,
                 expression_dim: int = 4,
                 hidden_dim: int = 128,
                 time_embedding_dim: int = 64):
        super().__init__()
        
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim
        self.midihum_dim = midihum_dim
        self.expression_dim = expression_dim
        self.time_embedding_dim = time_embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.categorical_encoder = ConditionEncoder(min(categorical_dim, 64), hidden_dim)
        self.continuous_encoder = ConditionEncoder(continuous_dim, hidden_dim)
        self.midihum_encoder = ConditionEncoder(midihum_dim, hidden_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.model = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expression_dim)
        )
        
        self.to(self.device)
        
        self.conditioning_provider = ExpressiveConditioningProvider(
            categorical_dim=categorical_dim,
            continuous_dim=continuous_dim,
            midihum_dim=midihum_dim,
            use_midihum=(midihum_dim > 0),
            device=self.device
        )
    
    def embed_time_parameter(self, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = get_timestep_embedding(timesteps, self.time_embedding_dim)
        time_features = self.time_mlp(time_emb)
        return time_features
    
    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor, 
                midihum: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        categorical_features = self.categorical_encoder(categorical)
        continuous_features = self.continuous_encoder(continuous)
        midihum_features = self.midihum_encoder(midihum)
        
        time_features = self.embed_time_parameter(t)
        
        fused_features = torch.cat([categorical_features, continuous_features, 
                                   midihum_features, time_features], dim=1)
        return self.model(fused_features)
    
    def _sample_trajectory(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x0 * (1 - t)[:, None]
    
    def _compute_vector_field(self, categorical: torch.Tensor, continuous: torch.Tensor, 
                             midihum: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(categorical, continuous, midihum, t)
    
    def flow_matching_loss(self, categorical: torch.Tensor, continuous: torch.Tensor, 
                           midihum: torch.Tensor, x0: torch.Tensor, 
                           xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        true_vector_field = -x0
        
        predicted_vector_field = self._compute_vector_field(categorical, continuous, midihum, t)
        
        return torch.mean((predicted_vector_field - true_vector_field) ** 2)
    
    def train(self, training_notes: List[List[ExpressiveNote]], feature_extractor, epochs: int = 1000, 
              batch_size: int = 32, lr: float=1e-3):
        processed_notes = self._validate_and_process_notes(training_notes, feature_extractor)
        
        data = self._prepare_data(processed_notes, feature_extractor)
    
        encoded_notes = []
        categoricals = []
        continuous = []
        midihums = []
        targets = []

        for piece_notes in training_notes:
            encoded_features = feature_extractor.encode_features(piece_notes, fit=True, use_midihum=self.conditioning_provider.use_midihum)
            encoded_tensor = torch.tensor(encoded_features, dtype=torch.float32, device=self.device)
            
            categorical_features = encoded_tensor[:, :self.conditioning_provider.categorical_conditioner.input_dim]
            continuous_features = encoded_tensor[:, self.conditioning_provider.categorical_conditioner.input_dim:]
            
            processed_categorical = self.conditioning_provider.categorical_conditioner.process(categorical_features)
            categoricals.append(processed_categorical)
            
            processed_continuous = self.conditioning_provider.continuous_conditioner.process(continuous_features)
            continuous.append(processed_continuous)
            

        data = self._prepare_data(encoded_notes, feature_extractor)

        categoricals = data['categorical']
        continuous = data['continuous']
        midihums = data.get('midihum')
        targets = data['targets']

        has_midihum = midihums is not None
        if has_midihum:
            dataset = TensorDataset(categoricals, continuous, midihums, targets)
        else:
            dataset = TensorDataset(categoricals, continuous, targets)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0

            for batch in dataloader:
                if has_midihum:
                    batch_categorical, batch_continuous, batch_midihum, batch_targets = batch
                else:
                    batch_categorical, batch_continuous, batch_targets = batch
                    batch_midihum = None
            
                t = torch.rand(batch_categorical.size(0), device=self.device)
                
                xt = self._sample_trajectory(batch_targets, t)
                
                optimizer.zero_grad()
                loss = self.flow_matching_loss(batch_categorical, batch_continuous, 
                                              batch_midihum, batch_targets, xt, t)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("Training completed!")
    
    def _validate_and_process_notes(self, training_notes: List[List[ExpressiveNote]], feature_extractor) -> List[np.ndarray]:
        processed_notes = []
        
        for piece_notes in training_notes:
            try:
                encoded_features = feature_extractor.encode_features(piece_notes, fit=True, use_midihum=self.conditioning_provider.use_midihum)
                processed_notes.append(encoded_features)
            except ValueError as e:
                if "inhomogeneous shape" in str(e):
                    processed_features = self._handle_inhomogeneous_features(piece_notes, feature_extractor)
                    processed_notes.append(processed_features)
                else:
                    raise
        
        return processed_notes
    
    def _handle_inhomogeneous_features(self, notes: List[ExpressiveNote], feature_extractor) -> np.ndarray:
        categorical_features = []
        continuous_features = []
        
        for note in notes:
            rhythmic_context = note.rhythmic_context or "boundary"
            ir_label = note.ir_label or "boundary"
            
            rhythmic_context_idx = self._map_rhythmic_context(rhythmic_context)
            ir_label_idx = self._map_ir_label(ir_label)
            
            categorical_features.append([rhythmic_context_idx, ir_label_idx])
            pitch = note.pitch or 0
            onset_beat = note.onset_beat or 0
            duration_beat = note.duration_beat or 0
            
            continuous_features.append([pitch, onset_beat, duration_beat])
        
        categorical_array = np.array(categorical_features)
        continuous_array = np.array(continuous_features)
        
        if categorical_array.ndim == 1:
            categorical_array = categorical_array.reshape(-1, 1)
        if continuous_array.ndim == 1:
            continuous_array = continuous_array.reshape(-1, 1)
        
        try:
            all_features = np.hstack([categorical_array, continuous_array])
        except ValueError as e:
            print(f"error: {e}")
            print(f"categorical array: {categorical_array.shape}")
            print(f"continuous: {continuous_array.shape}")
            all_features = self._zero_pad_features(categorical_array, continuous_array)
        
        return all_features
    
    def _zero_pad_features(self, categorical: np.ndarray, continuous: np.ndarray) -> np.ndarray:
        rows = max(categorical.shape[0], continuous.shape[0])
        
        padded_categorical = np.zeros((rows, categorical.shape[1]))
        padded_continuous = np.zeros((rows, continuous.shape[1]))
        
        padded_categorical[:categorical.shape[0], :categorical.shape[1]] = categorical
        padded_continuous[:continuous.shape[0], :continuous.shape[1]] = continuous
        
        return np.hstack([padded_categorical, padded_continuous])
    
    def _map_rhythmic_context(self, context: str) -> int:
        mapping = {"s-s-l": 0, "s-l-s": 1, "l-s-s": 2, "boundary": 3,
                   "s-l-l": 4, "l-l-s": 5, "l-s-l": 6}
        return mapping.get(context, 3)
    
    def _map_ir_label(self, label: str) -> int:
        mapping = {"Process": 0, "Reversal": 1, "Registral_Return": 2, 
                   "Intervallic_Duplication": 3, "boundary": 4}
        return mapping.get(label, 4)
    
    def _prepare_data(self, training_notes: List[np.ndarray], feature_extractor) -> Dict[str, torch.Tensor]:
        categoricals = []
        continuous = []
        midihums = []
        targets = []
        
        for encoded_features in training_notes:
            if not isinstance(encoded_features, np.ndarray):
                encoded_features = np.array(encoded_features)
            
            categorical_dim = self.conditioning_provider.categorical_conditioner.input_dim
            continuous_dim = encoded_features.shape[1] - categorical_dim
            
            if encoded_features.shape[1] < categorical_dim:
                categorical_features = np.zeros((encoded_features.shape[0], categorical_dim))
                continuous_features = encoded_features
            else:
                categorical_features = encoded_features[:, :categorical_dim]
                continuous_features = encoded_features[:, categorical_dim:]
            
            categorical_tensor = torch.tensor(categorical_features, dtype=torch.float32, device=self.device)
            continuous_tensor = torch.tensor(continuous_features, dtype=torch.float32, device=self.device)
            
            processed_categorical = self.conditioning_provider.categorical_conditioner.process(categorical_tensor)
            processed_continuous = self.conditioning_provider.continuous_conditioner.process(continuous_tensor)
            
            categoricals.append(processed_categorical)
            continuous.append(processed_continuous)
            
    def predict(self, expressive_notes: List[ExpressiveNote], conditions: Optional[Dict[str, List[List[float]]]] = None) -> List[ExpressiveNote]:
        if conditions is None:
            attributes = ConditioningAttributes(expressive_notes=expressive_notes)
            processed_features = self.conditioning_provider.process(attributes)
            categorical_tensor = processed_features['categorical']
            continuous_tensor = processed_features['continuous']
            midihum_tensor = processed_features['midihum']
        else:
            categorical_tensor = torch.tensor(conditions['categorical'], dtype=torch.float32, device=self.device)
            continuous_tensor = torch.tensor(conditions['continuous'], dtype=torch.float32, device=self.device)
            midihum_tensor = torch.tensor(conditions['midihum'], dtype=torch.float32, device=self.device)
        
        t = torch.zeros(categorical_tensor.size(0), device=self.device)
        
        with torch.no_grad():
            predictions = self.forward(categorical_tensor, continuous_tensor, midihum_tensor, t).cpu().numpy()
        
        predicted_notes = []
        for i, note in enumerate(expressive_notes):
            predicted_note = ExpressiveNote(
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
                beat_period=predictions[i, 0],
                timing=predictions[i, 1],
                velocity=int(round(predictions[i, 2] * 127)),
                articulation_log=predictions[i, 3]
            )
            predicted_notes.append(predicted_note)
        
        return predicted_notes
    
    def save(self, filepath: str) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'categorical_dim': self.categorical_dim,
            'continuous_dim': self.continuous_dim,
            'midihum_dim': self.midihum_dim,
            'expression_dim': self.expression_dim,
            'hidden_dim': self.model[0].in_features // 4,  
            'time_embedding_dim': self.time_embedding_dim
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ConditionalFlowMatchingModel':
        checkpoint = torch.load(filepath)
        model = cls(
            categorical_dim=checkpoint['categorical_dim'],
            continuous_dim=checkpoint['continuous_dim'],
            midihum_dim=checkpoint['midihum_dim'],
            expression_dim=checkpoint['expression_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            time_embedding_dim=checkpoint['time_embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class FMExpressiveModel:    
    def __init__(self, categorical_dim: int = 64, continuous_dim: int = 64, 
                 midihum_dim: int = 64, expression_dim: int = 4, 
                 hidden_dim: int = 128, time_embedding_dim: int = 64):
        self.model = ConditionalFlowMatchingModel(
            categorical_dim, continuous_dim, midihum_dim, expression_dim, 
            hidden_dim, time_embedding_dim
        )
    
    def train(self, training_notes: List[List[ExpressiveNote]], feature_extractor, 
              epochs: int = 1000, batch_size: int = 32, lr: float = 1e-3):
        self.model.train(training_notes, feature_extractor, epochs, batch_size, lr)
    
    def predict(self, expressive_notes: List[ExpressiveNote], conditions: Optional[Dict[str, List[List[float]]]] = None) -> List[ExpressiveNote]:
        return self.model.predict(expressive_notes, conditions)
    
    def save(self, filepath: str) -> None:
        self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        self.model = ConditionalFlowMatchingModel.load(filepath)