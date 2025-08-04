# PerformanceMOS - Performance Quality Assessment Extension for Aria

This repository extends the [Aria music generation model](./README.md) with regression capabilities for performance quality assessment. The PerformanceMOS model can evaluate musical performances and predict quality scores on a 0-100 scale.

## Prerequisites

- Complete Aria setup following the [main Aria README](./README.md)
- Python 3.11+
- CUDA-compatible GPU (recommended)

## Installation

After setting up the base Aria environment:

```bash
# Install additional dependencies for regression training and evaluation
pip install -e ".[all]"

# Additional packages for regression features
pip install scikit-learn matplotlib seaborn
```

## What's New in PerformanceMOS

- **Regression Training**: Train models to predict performance quality scores (0-100)
- **Performance Evaluation**: Evaluate on Vienna test set and custom MIDI collections
- **Quality Assessment**: Automated scoring of musical performances
- **YQX+ Evaluation**: PerformanceMOS serves as an evaluation method for YQX+ music generation research

## Model Architecture

- **Base Model**: TransformerREG (regression head on transformer)
- **Architecture**: 1536 d_model, 24 attention heads, 16 layers
- **Input**: Tokenized MIDI sequences (max 1024 tokens)
- **Output**: Performance score (0-100 range)
- **Training**: MSE loss, AdamW optimizer, linear LR schedule (1e-5)

## 📁 Directory Structure

```
end-to-end-regression/
├── dataset/                    # Put your datasets here
│   ├── YCU-PPE-III-Midi/      # YCU dataset with performance scores
│   └── augmented_performances/ # Augmented performances dataset
├── ckpt/                      # Pre-trained checkpoints
│   └── base.safetensors       # Base Aria model checkpoint
├── experiments/               # Training outputs (auto-created)
└── train.py                   # Training script

eval/
├── eval.py                    # Evaluation script
├── eval.bat                   # Windows evaluation batch script
└── vienna-test/               # Vienna test dataset

YQX_result/                    # YQX+ evaluation results and baselines
├── baseline-Dexter/           # Dexter baseline method outputs
├── baseline-virtuosoNet/      # VirtuosoNet baseline method outputs
└── New_output/                # YQX+ method variations
    ├── bvae_*/                # Beta-VAE based methods
    ├── flow_*/                # Flow-based methods  
    ├── gmm_*/                 # Gaussian Mixture Model methods
    └── xgboost_*/             # XGBoost-based methods
```

## 🚀 Quick Start

### 1. Setup Data

Put your datasets in the `end-to-end-regression/dataset/` folder. We provide:
- **YCU-PPE-III-Midi**: YCU dataset with performance quality scores (0-100)
- **augmented_performances**: Augmented performance data with quality corruptions， source dataset are ATEPP and ASAP

### 2. Get Pre-trained Checkpoint

Download the base Aria model checkpoint and place it at:
```
end-to-end-regression/ckpt/base.safetensors
```

### 3. Training

```bash
cd end-to-end-regression

# Train on YCU and augmented dataset
python train.py --dataset all --epochs 10 --batch_size 12

# Train on YCU dataset
python train.py --dataset ycu --epochs 10 --batch_size 12

# Train on augmented dataset  
python train.py --dataset augmented --epochs 10 --batch_size 12

# Custom training with different scheduler
python train.py --dataset ycu --epochs 15 --batch_size 8 --scheduler cosine
```

### 4. Evaluation

```bash
cd eval

# Evaluate on Vienna test set (default)
python eval.py --checkpoint_path ../end-to-end-regression/experiments/best_model.pt

# Evaluate on YCU test set
python eval.py --checkpoint_path ../end-to-end-regression/experiments/best_model.pt --datasets ycu

# Evaluate on custom MIDI folder
python eval.py --checkpoint_path ../end-to-end-regression/experiments/best_model.pt --custom_midi_dir /path/to/midi/files

# Windows users can use eval.bat for comprehensive YQX+ evaluation
eval.bat
```

## 🎼 YQX+ Evaluation

PerformanceMOS is used as an evaluation method for YQX+ music generation research. The `YQX_result/` folder contains generated outputs from various methods that can be evaluated using PerformanceMOS.

### YQX+ Methods and Baselines

**Baseline Methods:**
- **Dexter**
- **VirtuosoNet**

**YQX+ Method Variations:**
- **Beta-VAE (bvae_*)**
- **Flow Models (flow_*)**
- **Gaussian Mixture Models (gmm_*)**
- **XGBoost (xgboost_*)**

### Batch Evaluation with eval.bat

The `eval.bat` script automatically evaluates all YQX+ methods and baselines:

```bash
# Runs PerformanceMOS evaluation on:
# 1. Vienna test set (default evaluation)
# 2. All baseline methods (Dexter, VirtuosoNet)  
# 3. All YQX+ method variations in New_output/

cd eval
eval.bat
```

Each method folder is evaluated separately, generating individual detailed CSV files with PerformanceMOS quality scores for comparative analysis.

## ⚙️ Configuration Options

### Training Parameters (`train.py`)

- `--dataset`: Dataset to use for training
  - `ycu`: YCU-PPE-III dataset only
  - `augmented`: Augmented performances dataset only
  - `all`: both YCU and Augmented dataset
  - **Default**: `all`
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 12)
- `--scheduler`: Learning rate scheduler (default: linear)
  - `linear`: Linear decay to 0
  - `poly`: Polynomial decay
  - `cosine`: Cosine annealing
  - `constant`: No decay
- `--checkpoint`: Path to pre-trained model (default: ./ckpt/base.safetensors)
- `--max_seq_len`: Maximum sequence length (default: 1024)
- `--project_dir`: Output directory (default: ./experiments)

### Evaluation Parameters (`eval.py`)

- `--checkpoint_path`: Path to trained model checkpoint (**required**)
- `--datasets`: Datasets to evaluate (default: ["vienna"])
  - `vienna`: Vienna test dataset (score-based MIDI files)
  - `ycu`: YCU test split
- `--custom_midi_dir`: Path to custom MIDI folder for inference
- `--max_seq_len`: Maximum sequence length (default: 1024)

## 📊 Output Files

### Training Outputs
- `./experiments/best_model.pt` - Best model based on validation R² score
- `./experiments/logs.txt` - Training logs
- `./experiments/training_results.json` - Training metrics summary

### Evaluation Outputs
The evaluation script generates a detailed CSV file with predicted scores for each MIDI file:

**CSV Structure:**
```csv
filename,predicted_score,dataset
file1.mid,85.2,vienna
file2.mid,60.0,ycu
...
```

## 🔧 Dataset Information

### YCU-PPE-III Dataset
- **Source**: YCU Performance Assessment Dataset
- **Score Range**: 0-100 (continuous performance quality scores)
- **Content**: Classical piano performances with human expert ratings

### Augmented Performances Dataset  
- **Source**: Augmented from ATEPP ASAP dataset
- **Score Range**: 0-100 (performance scores with simulated quality corruptions)
- **Content**: Various musical performances with systematic quality degradations

### Vienna 4X22 Test Set
- **Content**: Score-based MIDI files for inference testing, include both human performance and score midi
- **Purpose**: To support YQX+ evaluation, as a comparison

## 📖 Citation

[Publication citation placeholder - to be added upon paper acceptance]

## License

This extension follows the original Aria licensing under the Apache-2.0 license. See [LICENSE](../LICENSE) for details.

