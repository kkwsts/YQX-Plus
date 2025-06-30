"""
Simple demonstration of JASCO-style Flow Matching for Expressive Performance
"""

import torch
import numpy as np
from flow_jasco import JASCOExpressiveModel, JASCOConfig, get_pretrained_jasco_expressive
from features import FeatureExtractor
from expressivenote import ExpressiveNote


def create_demo_notes(n_notes: int = 50) -> list:
    """Create demo notes for testing"""
    notes = []
    for i in range(n_notes):
        note = ExpressiveNote(
            pitch=60 + (i % 12),  # C4 to B4
            onset_beat=i * 0.5,
            duration_beat=0.25 + np.random.random() * 0.5,
            voice=0,
            pitch_interval=np.random.randint(-12, 13),
            duration_ratio=0.5 + np.random.random() * 1.0,
            rhythmic_context="quarter",
            ir_label="T",
            ir_closure=False,
            position_in_phrase=i % 8,
            # Targets
            beat_period=0.5 + np.random.random() * 0.5,
            timing=np.random.random() * 0.2 - 0.1,
            velocity=80 + np.random.randint(-20, 21),
            articulation_log=np.random.random() * 0.5 - 0.25
        )
        notes.append(note)
    return notes


def demo_jasco():
    """Demonstrate JASCO-style flow matching"""
    print("JASCO-Style Flow Matching Demo")
    print("=" * 40)
    
    # Create JASCO configuration (following original structure)
    config = JASCOConfig(
        dim=128,           # transformer dimension
        num_heads=8,       # attention heads
        flow_dim=4,        # expression parameters (beat_period, timing, velocity, articulation)
        num_layers=6,      # transformer layers
        hidden_scale=4,    # feedforward scale
        learning_rate=1e-4,
        warmup_steps=500,
        # CFG parameters (following original JASCO)
        cfg_coef_all=3.0,
        cfg_coef_txt=1.0,
        euler_steps=50,
        ode_rtol=1e-5,
        ode_atol=1e-5
    )
    
    print(f"Configuration:")
    print(f"  Transformer dim: {config.dim}")
    print(f"  Flow dim: {config.flow_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    
    # Create model
    model = JASCOExpressiveModel(config, use_midihum_features=False)
    print(f"\nUsing device: {model.device}")
    
    # Create demo data
    print("\nCreating demo data...")
    training_pieces = [create_demo_notes(30) for _ in range(3)]  # 3 pieces
    test_notes = create_demo_notes(10)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Train model
    print("\nTraining JASCO model...")
    model.train(
        training_pieces, 
        feature_extractor, 
        epochs=50,  # Quick demo
        batch_size=8
    )
    
    # Set generation parameters (following original JASCO API)
    model.set_generation_params(
        cfg_coef_all=2.0,
        cfg_coef_txt=0.5,
        euler=True,  # Use Euler for speed
        euler_steps=20
    )
    
    # Test prediction
    print("\nGenerating predictions...")
    predictions = model.predict(test_notes, feature_extractor)
    
    # Show results
    print("\nPrediction Results:")
    print("-" * 60)
    print(f"{'Note':<4} {'BP_true':<8} {'BP_pred':<8} {'Tim_true':<9} {'Tim_pred':<9} {'Vel_true':<8} {'Vel_pred':<8}")
    print("-" * 60)
    
    for i in range(len(test_notes)):
        true_note = test_notes[i]
        pred_note = predictions[i]
        
        print(f"{i+1:<4} {true_note.beat_period:<8.3f} {pred_note.beat_period:<8.3f} "
              f"{true_note.timing:<9.3f} {pred_note.timing:<9.3f} "
              f"{true_note.velocity:<8} {pred_note.velocity:<8}")
    
    # Test save/load
    print("\nTesting save/load...")
    model.save("demo_jasco_model.pt")
    
    # Load model
    new_model = get_pretrained_jasco_expressive()
    new_model.load("demo_jasco_model.pt")
    
    # Verify loaded model works
    new_predictions = new_model.predict(test_notes[:3], feature_extractor)
    print(f"Loaded model predictions: {len(new_predictions)} notes")
    
    print("\nDemo completed successfully!")
    
    # Clean up
    import os
    if os.path.exists("demo_jasco_model.pt"):
        os.remove("demo_jasco_model.pt")


def demo_cfg_modes():
    """Demonstrate different CFG modes"""
    print("\nCFG Modes Demo")
    print("=" * 20)
    
    config = JASCOConfig(dim=64, num_layers=4, flow_dim=4)
    model = JASCOExpressiveModel(config, use_midihum_features=False)
    
    # Quick training
    training_pieces = [create_demo_notes(20)]
    feature_extractor = FeatureExtractor()
    model.train(training_pieces, feature_extractor, epochs=20, batch_size=4)
    
    test_notes = create_demo_notes(3)
    
    # Test different CFG settings
    cfg_settings = [
        {"cfg_coef_all": 0.0, "cfg_coef_txt": 0.0, "name": "Unconditional"},
        {"cfg_coef_all": 1.0, "cfg_coef_txt": 0.0, "name": "All Conditions"},
        {"cfg_coef_all": 0.0, "cfg_coef_txt": 1.0, "name": "Text Only"},
        {"cfg_coef_all": 2.0, "cfg_coef_txt": 0.5, "name": "Mixed CFG"},
    ]
    
    for setting in cfg_settings:
        model.set_generation_params(**{k: v for k, v in setting.items() if k != "name"})
        predictions = model.predict(test_notes, feature_extractor)
        
        avg_bp = np.mean([p.beat_period for p in predictions])
        avg_timing = np.mean([p.timing for p in predictions])
        
        print(f"{setting['name']}: BP={avg_bp:.3f}, Timing={avg_timing:.3f}")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demos
    demo_jasco()
    demo_cfg_modes() 