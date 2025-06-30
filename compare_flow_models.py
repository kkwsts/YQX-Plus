"""
Comparison script between original Flow Matching and JASCO-style Flow Matching
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

from flow import ConditionalFlowModel, FlowConfig
from flow_jasco import JASCOExpressiveModel, FlowMatchingConfig
from features import FeatureExtractor
from expressivenote import ExpressiveNote


def create_test_data(n_notes: int = 200):
    """Create test data for comparison"""
    notes = []
    for i in range(n_notes):
        # Create more realistic musical patterns
        pitch = 60 + (i % 12) + (i // 12) % 12  # Walking pattern
        onset = i * 0.25 + np.random.normal(0, 0.02)  # Slight timing variations
        
        note = ExpressiveNote(
            pitch=pitch,
            onset_beat=onset,
            duration_beat=0.2 + np.random.exponential(0.3),
            voice=0,
            pitch_interval=pitch - (60 + ((i-1) % 12) + ((i-1) // 12) % 12) if i > 0 else 0,
            duration_ratio=0.5 + np.random.gamma(2, 0.25),
            rhythmic_context=["eighth", "quarter", "half"][i % 3],
            ir_label=["T", "S", "D"][i % 3],
            ir_closure=i % 8 == 7,
            position_in_phrase=i % 8,
            # Targets with some musical logic
            beat_period=0.4 + 0.2 * np.sin(i * 0.1) + np.random.normal(0, 0.05),
            timing=0.1 * np.sin(i * 0.2) + np.random.normal(0, 0.02),
            velocity=int(80 + 20 * np.sin(i * 0.15) + np.random.normal(0, 5)),
            articulation_log=-0.2 + 0.3 * np.cos(i * 0.12) + np.random.normal(0, 0.1)
        )
        notes.append(note)
    
    return notes


def compare_models():
    """Compare original Flow Matching with JASCO-style implementation"""
    print("Comparing Flow Matching Models")
    print("=" * 60)
    
    # Create test data
    print("Creating test data...")
    all_notes = create_test_data(300)
    
    # Split data
    train_notes = [all_notes[:200]]  # Single piece for training
    test_notes = all_notes[200:]
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    print(f"Training notes: {len(train_notes[0])}")
    print(f"Test notes: {len(test_notes)}")
    
    # =============================================================================
    # Original Flow Matching Model
    # =============================================================================
    print("\n" + "="*60)
    print("ORIGINAL FLOW MATCHING MODEL")
    print("="*60)
    
    original_config = FlowConfig(
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        learning_rate=1e-4,
        epochs=200,
        batch_size=16
    )
    
    original_model = ConditionalFlowModel(original_config, use_midihum_features=False)
    
    print("Training original model...")
    start_time = time.time()
    original_model.train(train_notes, feature_extractor)
    original_train_time = time.time() - start_time
    
    print("Testing original model...")
    start_time = time.time()
    original_predictions = original_model.predict(test_notes, feature_extractor)
    original_inference_time = time.time() - start_time
    
    # =============================================================================
    # JASCO-style Flow Matching Model
    # =============================================================================
    print("\n" + "="*60)
    print("JASCO-STYLE FLOW MATCHING MODEL")
    print("="*60)
    
    jasco_config = FlowMatchingConfig(
        d_model=256,
        num_layers=8,
        num_heads=8,
        d_ff=1024,
        learning_rate=1e-4,
        warmup_steps=500,
        use_weighted_loss=True,
        dropout=0.1
    )
    
    jasco_model = JASCOExpressiveModel(jasco_config, use_midihum_features=False)
    
    print("Training JASCO model...")
    start_time = time.time()
    jasco_model.train(train_notes, feature_extractor, epochs=200, batch_size=16)
    jasco_train_time = time.time() - start_time
    
    print("Testing JASCO model...")
    start_time = time.time()
    jasco_predictions = jasco_model.predict(test_notes, feature_extractor)
    jasco_inference_time = time.time() - start_time
    
    # =============================================================================
    # Comparison and Analysis
    # =============================================================================
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Calculate metrics for both models
    def calculate_metrics(true_notes, pred_notes, model_name):
        bp_mae = np.mean([abs(t.beat_period - p.beat_period) for t, p in zip(true_notes, pred_notes)])
        timing_mae = np.mean([abs(t.timing - p.timing) for t, p in zip(true_notes, pred_notes)])
        velocity_mae = np.mean([abs(t.velocity - p.velocity) for t, p in zip(true_notes, pred_notes)])
        art_mae = np.mean([abs(t.articulation_log - p.articulation_log) for t, p in zip(true_notes, pred_notes)])
        
        bp_rmse = np.sqrt(np.mean([(t.beat_period - p.beat_period)**2 for t, p in zip(true_notes, pred_notes)]))
        timing_rmse = np.sqrt(np.mean([(t.timing - p.timing)**2 for t, p in zip(true_notes, pred_notes)]))
        velocity_rmse = np.sqrt(np.mean([(t.velocity - p.velocity)**2 for t, p in zip(true_notes, pred_notes)]))
        art_rmse = np.sqrt(np.mean([(t.articulation_log - p.articulation_log)**2 for t, p in zip(true_notes, pred_notes)]))
        
        return {
            'mae': {'bp': bp_mae, 'timing': timing_mae, 'velocity': velocity_mae, 'articulation': art_mae},
            'rmse': {'bp': bp_rmse, 'timing': timing_rmse, 'velocity': velocity_rmse, 'articulation': art_rmse}
        }
    
    original_metrics = calculate_metrics(test_notes, original_predictions, "Original")
    jasco_metrics = calculate_metrics(test_notes, jasco_predictions, "JASCO")
    
    # Print comparison table
    print("\nPERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Original MAE':<12} {'JASCO MAE':<12} {'Original RMSE':<13} {'JASCO RMSE':<13}")
    print("-" * 80)
    
    metrics_names = ['Beat Period', 'Timing', 'Velocity', 'Articulation']
    metrics_keys = ['bp', 'timing', 'velocity', 'articulation']
    
    for name, key in zip(metrics_names, metrics_keys):
        orig_mae = original_metrics['mae'][key]
        jasco_mae = jasco_metrics['mae'][key]
        orig_rmse = original_metrics['rmse'][key]
        jasco_rmse = jasco_metrics['rmse'][key]
        
        print(f"{name:<15} {orig_mae:<12.4f} {jasco_mae:<12.4f} {orig_rmse:<13.4f} {jasco_rmse:<13.4f}")
    
    print("\nTIMING COMPARISON:")
    print("-" * 50)
    print(f"Training Time:")
    print(f"  Original: {original_train_time:.2f} seconds")
    print(f"  JASCO:    {jasco_train_time:.2f} seconds")
    print(f"  Speedup:  {original_train_time/jasco_train_time:.2f}x")
    
    print(f"\nInference Time:")
    print(f"  Original: {original_inference_time:.4f} seconds")
    print(f"  JASCO:    {jasco_inference_time:.4f} seconds")
    print(f"  Speedup:  {original_inference_time/jasco_inference_time:.2f}x")
    
    # Model complexity comparison
    def count_parameters(model):
        if hasattr(model, 'model') and model.model is not None:
            return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        return 0
    
    original_params = count_parameters(original_model)
    jasco_params = count_parameters(jasco_model)
    
    print(f"\nMODEL COMPLEXITY:")
    print(f"  Original parameters: {original_params:,}")
    print(f"  JASCO parameters:    {jasco_params:,}")
    print(f"  Parameter ratio:     {jasco_params/original_params:.2f}x")
    
    # =============================================================================
    # Visualization
    # =============================================================================
    print("\nCreating visualizations...")
    
    # Plot predictions vs ground truth
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Predictions vs Ground Truth', fontsize=16)
    
    # Select first 50 notes for visualization
    viz_notes = min(50, len(test_notes))
    
    # Beat Period
    axes[0, 0].plot([n.beat_period for n in test_notes[:viz_notes]], 'k-', label='Ground Truth', linewidth=2)
    axes[0, 0].plot([n.beat_period for n in original_predictions[:viz_notes]], 'b--', label='Original', alpha=0.7)
    axes[0, 0].plot([n.beat_period for n in jasco_predictions[:viz_notes]], 'r:', label='JASCO', alpha=0.7)
    axes[0, 0].set_title('Beat Period')
    axes[0, 0].set_ylabel('Beat Period (s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Timing
    axes[0, 1].plot([n.timing for n in test_notes[:viz_notes]], 'k-', label='Ground Truth', linewidth=2)
    axes[0, 1].plot([n.timing for n in original_predictions[:viz_notes]], 'b--', label='Original', alpha=0.7)
    axes[0, 1].plot([n.timing for n in jasco_predictions[:viz_notes]], 'r:', label='JASCO', alpha=0.7)
    axes[0, 1].set_title('Timing')
    axes[0, 1].set_ylabel('Timing (s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity
    axes[1, 0].plot([n.velocity for n in test_notes[:viz_notes]], 'k-', label='Ground Truth', linewidth=2)
    axes[1, 0].plot([n.velocity for n in original_predictions[:viz_notes]], 'b--', label='Original', alpha=0.7)
    axes[1, 0].plot([n.velocity for n in jasco_predictions[:viz_notes]], 'r:', label='JASCO', alpha=0.7)
    axes[1, 0].set_title('Velocity')
    axes[1, 0].set_ylabel('Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Articulation
    axes[1, 1].plot([n.articulation_log for n in test_notes[:viz_notes]], 'k-', label='Ground Truth', linewidth=2)
    axes[1, 1].plot([n.articulation_log for n in original_predictions[:viz_notes]], 'b--', label='Original', alpha=0.7)
    axes[1, 1].plot([n.articulation_log for n in jasco_predictions[:viz_notes]], 'r:', label='JASCO', alpha=0.7)
    axes[1, 1].set_title('Articulation (log)')
    axes[1, 1].set_ylabel('Articulation (log)')
    axes[1, 1].set_xlabel('Note Index')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flow_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Error Distributions', fontsize=16)
    
    # Calculate errors
    orig_bp_errors = [t.beat_period - p.beat_period for t, p in zip(test_notes, original_predictions)]
    jasco_bp_errors = [t.beat_period - p.beat_period for t, p in zip(test_notes, jasco_predictions)]
    
    orig_timing_errors = [t.timing - p.timing for t, p in zip(test_notes, original_predictions)]
    jasco_timing_errors = [t.timing - p.timing for t, p in zip(test_notes, jasco_predictions)]
    
    orig_vel_errors = [t.velocity - p.velocity for t, p in zip(test_notes, original_predictions)]
    jasco_vel_errors = [t.velocity - p.velocity for t, p in zip(test_notes, jasco_predictions)]
    
    orig_art_errors = [t.articulation_log - p.articulation_log for t, p in zip(test_notes, original_predictions)]
    jasco_art_errors = [t.articulation_log - p.articulation_log for t, p in zip(test_notes, jasco_predictions)]
    
    # Plot histograms
    axes[0, 0].hist(orig_bp_errors, bins=20, alpha=0.7, label='Original', color='blue')
    axes[0, 0].hist(jasco_bp_errors, bins=20, alpha=0.7, label='JASCO', color='red')
    axes[0, 0].set_title('Beat Period Errors')
    axes[0, 0].set_xlabel('Error (s)')
    axes[0, 0].legend()
    
    axes[0, 1].hist(orig_timing_errors, bins=20, alpha=0.7, label='Original', color='blue')
    axes[0, 1].hist(jasco_timing_errors, bins=20, alpha=0.7, label='JASCO', color='red')
    axes[0, 1].set_title('Timing Errors')
    axes[0, 1].set_xlabel('Error (s)')
    axes[0, 1].legend()
    
    axes[1, 0].hist(orig_vel_errors, bins=20, alpha=0.7, label='Original', color='blue')
    axes[1, 0].hist(jasco_vel_errors, bins=20, alpha=0.7, label='JASCO', color='red')
    axes[1, 0].set_title('Velocity Errors')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].legend()
    
    axes[1, 1].hist(orig_art_errors, bins=20, alpha=0.7, label='Original', color='blue')
    axes[1, 1].hist(jasco_art_errors, bins=20, alpha=0.7, label='JASCO', color='red')
    axes[1, 1].set_title('Articulation Errors')
    axes[1, 1].set_xlabel('Error (log)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('error_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nComparison completed!")
    print("Plots saved as 'flow_models_comparison.png' and 'error_distributions.png'")


def analyze_architectural_differences():
    """Analyze and explain the architectural differences"""
    print("\n" + "="*60)
    print("ARCHITECTURAL DIFFERENCES ANALYSIS")
    print("="*60)
    
    print("\n1. FLOW MATCHING FORMULATION:")
    print("   Original: Basic conditional flow matching")
    print("   JASCO:    Optimal Transport (OT) flow matching with sigma_min")
    print("             - More stable training")
    print("             - Better numerical properties")
    
    print("\n2. NEURAL NETWORK ARCHITECTURE:")
    print("   Original: Simple MLP with skip connections")
    print("   JASCO:    Transformer with U-Net style connections")
    print("             - Self-attention for long-range dependencies")
    print("             - ALiBi positional bias")
    print("             - Convolutional positional encoding")
    
    print("\n3. CONDITIONING MECHANISM:")
    print("   Original: Direct concatenation of features")
    print("   JASCO:    Information bottleneck + temporal blurring")
    print("             - Reduces overfitting to irrelevant features")
    print("             - Better generalization")
    
    print("\n4. TRAINING ENHANCEMENTS:")
    print("   Original: Standard MSE loss")
    print("   JASCO:    Weighted loss (1+t) + warmup + gradient clipping")
    print("             - More stable training")
    print("             - Better convergence")
    
    print("\n5. INFERENCE CAPABILITIES:")
    print("   Original: Simple Euler integration")
    print("   JASCO:    Support for classifier-free guidance")
    print("             - Multi-source CFG for better control")
    print("             - More sophisticated sampling")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comparison
    compare_models()
    analyze_architectural_differences()