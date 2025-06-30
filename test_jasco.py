"""
Test script for JASCO-style Flow Matching model
"""

import torch
import numpy as np
from pathlib import Path

from flow_jasco import JASCOExpressiveModel, JASCOConfig
from features import FeatureExtractor
from expressivenote import ExpressiveNote
from data_loader import DataLoader


def create_sample_notes(n_notes: int = 100) -> list:
    """Create sample notes for testing"""
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


def test_jasco_model():
    """Test the JASCO flow matching model"""
    print("Testing JASCO Flow Matching Model")
    print("=" * 50)
    
    # Create configuration
    config = JASCOConfig(
        dim=256,  # Smaller for testing
        num_layers=8,  # Fewer layers for testing
        num_heads=8,
        flow_dim=4,  # 4 expression parameters
        hidden_scale=4,
        learning_rate=1e-4,
        warmup_steps=1000
    )
    
    # Create sample data
    print("Creating sample data...")
    training_pieces = [create_sample_notes(50) for _ in range(5)]  # 5 pieces, 50 notes each
    test_notes = create_sample_notes(20)
    
    # Initialize feature extractor and model
    print("Initializing model...")
    feature_extractor = FeatureExtractor()
    model = JASCOExpressiveModel(config, use_midihum_features=True)
    
    print(f"Using device: {model.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Train model
    print("\nTraining model...")
    model.train(
        training_pieces, 
        feature_extractor, 
        epochs=100,  # Fewer epochs for testing
        batch_size=16
    )
    
    # Test prediction
    print("\nTesting prediction...")
    predictions = model.predict(test_notes, feature_extractor)
    
    # Compare predictions with ground truth
    print("\nComparison of predictions vs ground truth:")
    print("-" * 80)
    print(f"{'Note':<4} {'BP_true':<8} {'BP_pred':<8} {'Tim_true':<9} {'Tim_pred':<9} {'Vel_true':<8} {'Vel_pred':<8} {'Art_true':<8} {'Art_pred':<8}")
    print("-" * 80)
    
    for i in range(min(10, len(test_notes))):  # Show first 10 notes
        true_note = test_notes[i]
        pred_note = predictions[i]
        
        print(f"{i+1:<4} {true_note.beat_period:<8.3f} {pred_note.beat_period:<8.3f} "
              f"{true_note.timing:<9.3f} {pred_note.timing:<9.3f} "
              f"{true_note.velocity:<8} {pred_note.velocity:<8} "
              f"{true_note.articulation_log:<8.3f} {pred_note.articulation_log:<8.3f}")
    
    # Calculate metrics
    bp_mae = np.mean([abs(t.beat_period - p.beat_period) for t, p in zip(test_notes, predictions)])
    timing_mae = np.mean([abs(t.timing - p.timing) for t, p in zip(test_notes, predictions)])
    velocity_mae = np.mean([abs(t.velocity - p.velocity) for t, p in zip(test_notes, predictions)])
    art_mae = np.mean([abs(t.articulation_log - p.articulation_log) for t, p in zip(test_notes, predictions)])
    
    print("\nMean Absolute Errors:")
    print(f"Beat Period: {bp_mae:.4f}")
    print(f"Timing: {timing_mae:.4f}")
    print(f"Velocity: {velocity_mae:.4f}")
    print(f"Articulation: {art_mae:.4f}")
    
    # Test save/load
    print("\nTesting save/load...")
    model_path = "test_jasco_model.pt"
    model.save(model_path)
    
    # Create new model and load
    new_model = JASCOExpressiveModel(config, use_midihum_features=True)
    new_model.load(model_path)
    
    # Test prediction with loaded model
    new_predictions = new_model.predict(test_notes[:5], feature_extractor)
    
    # Check if predictions are identical
    identical = all(
        abs(p1.beat_period - p2.beat_period) < 1e-6 and
        abs(p1.timing - p2.timing) < 1e-6 and
        abs(p1.velocity - p2.velocity) < 1e-6 and
        abs(p1.articulation_log - p2.articulation_log) < 1e-6
        for p1, p2 in zip(predictions[:5], new_predictions)
    )
    
    print(f"Save/Load test: {'PASSED' if identical else 'FAILED'}")
    
    # Clean up
    Path(model_path).unlink(missing_ok=True)
    
    print("\nJASCO Flow Matching test completed!")


def test_real_data():
    """Test with real data if available"""
    print("\nTesting with real data...")
    
    try:
        # Try to load real data
        data_loader = DataLoader()
        if hasattr(data_loader, 'pieces') and data_loader.pieces:
            print(f"Found {len(data_loader.pieces)} pieces in dataset")
            
            # Use first few pieces for testing
            test_pieces = data_loader.pieces[:3] if len(data_loader.pieces) >= 3 else data_loader.pieces
            
            config = JASCOConfig(
                dim=128,  # Even smaller for real data test
                num_layers=6,
                num_heads=4,
                flow_dim=4,
                hidden_scale=2,
                learning_rate=5e-4,
                warmup_steps=500
            )
            
            feature_extractor = FeatureExtractor()
            model = JASCOExpressiveModel(config, use_midihum_features=False)  # Without midihum for speed
            
            # Split data
            train_pieces = test_pieces[:-1] if len(test_pieces) > 1 else test_pieces
            test_piece = test_pieces[-1]
            
            print(f"Training on {len(train_pieces)} pieces...")
            model.train(train_pieces, feature_extractor, epochs=50, batch_size=8)
            
            # Test on a subset of the test piece
            test_notes = test_piece[:20] if len(test_piece) > 20 else test_piece
            predictions = model.predict(test_notes, feature_extractor)
            
            print(f"Successfully predicted {len(predictions)} notes from real data")
            
            # Show some statistics
            if all(note.beat_period is not None for note in test_notes):
                bp_mae = np.mean([abs(t.beat_period - p.beat_period) for t, p in zip(test_notes, predictions)])
                print(f"Beat Period MAE: {bp_mae:.4f}")
            
        else:
            print("No real data available, skipping real data test")
            
    except Exception as e:
        print(f"Real data test failed: {e}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_jasco_model()
    test_real_data() 