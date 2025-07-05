# YQX+: Benchmarking Probabilistic Modelling of Performance Expression

This repository expand on the idea of the [YQX system]() for expressive music performance, that 
The system predicts expressive performance parameters from musical score context using a Bayesian model trained on human performances.


## Overview

- Extracts musical context features from MusicXML scores
- Trains a probabilistic expressive model on aligned score-performance data
- Predicts expressive parameters for new scores
- Renders expressive MIDI performances
- Automated testing framework with organized experiment tracking

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- partitura
- pretty_midi
- tqdm

<!-- Install dependencies with:

```bash
pip install -r requirements.txt
``` -->

## Usage

### Training Models

```bash
# Train GMM with different components
python yqx.py train.enabled=true model.type=gmm model.gmm.n_components=48

# Train with MIDIHUM features
python yqx.py train.enabled=true model.use_midihum_features=true

# Train with ASAP dataset
python yqx.py train.enabled=true data.use_asap=true
```


```bash
# Train and test automatically
python yqx.py train.enabled=true test.enabled=true model.gmm.n_components=32

# Test existing model
python yqx.py test.enabled=true model.gmm.n_components=48
```

### Rendering Single Performance

```bash
python yqx.py render.enabled=true render.input_score=<path_to_musicxml> render.output_midi=<output_midi_path> render.initial_tempo=<initial_tempo>
```

### Custom Experiments

```bash
# Custom experiment with name
python yqx.py train.enabled=true test.enabled=true model.gmm.n_components=64 model.use_midihum_features=true output.experiment_name=large_gmm_midihum
```

## Configuration

### Model Configuration

**GMM Model:**
- `model.type`: "gmm"
- `model.gmm.n_components`: Number of Gaussian components (default: 48)
- `model.gmm.random_state`: Random seed (default: 42)


**General Model Options:**
- `model.use_midihum_features`: Include MIDIHUM features (default: false)

### Training Configuration

- `train.enabled`: Enable training (default: false)
- `train.batch_size`: Training batch size (default: 32)
- `train.num_epochs`: Number of training epochs (default: 100)
- `train.learning_rate`: Learning rate (default: 0.001)

### Testing Configuration

- `test.enabled`: Enable Vienna4x22 testing (default: false)

### Output Configuration

- `output.experiment_name`: Custom experiment name (default: auto-generated)
- `output.include_model_params`: Include model params in filename (default: true)
- `output.artifacts_dir`: Artifacts directory (default: "artifacts")
- `output.ckpt_dir`: Model checkpoints directory (default: "ckpts")
- `output.output_dir`: Output directory (default: "outputs")

## File Organization

### Model Files
Models are automatically named with descriptive parameters:
- `gmm_nc48.pkl` - GMM with 48 components
- `gmm_nc32_midihum.pkl` - GMM with 32 components + MIDIHUM features
- `flow_hd256.pkl` - Flow model with 256 hidden dimensions
- `custom_name_gmm_nc64.pkl` - Custom experiment name

### Test Results
Testing creates organized output directories:
```
outputs/
├── gmm_nc32/
│   ├── Chopin_op10_no3_predicted.mid
│   ├── Chopin_op38_predicted.mid
│   ├── Mozart_K331_1st-mov_predicted.mid
│   ├── Schubert_D783_no15_predicted.mid
│   ├── *_predictions.png
│   └── experiment_summary.json
├── gmm_nc48_midihum/
└── flow_hd128/
```

## Vienna4x22 Testing

The system automatically tests on 4 Vienna4x22 pieces with designated tempos:
- Chopin_op10_no3: 30 BPM
- Chopin_op38: 120 BPM  
- Mozart_K331_1st-mov: 120 BPM
- Schubert_D783_no15: 130 BPM

Each test generates:
- Predicted MIDI files for each piece
- Prediction visualization plots
- Complete experiment metadata in JSON format

## Features 

Features from original YQX system:
- IR Category
- IR label
- Rhythmic context
- pitch_interval
- ir_closure
- position_in_phrase

#### MIDIHUM Features

We took the features from the [MIDIHUM](https://github.com/erwald/midihum) library and added them to the ExpressiveNote class.

## File Structure

- `yqx.py`: Main system implementation and CLI
- `features.py`: Feature extraction
- `expressivenote.py`: Expressive note data structure
- `gmm.py`: Bayesian expressive model (GMM-based)
- `flow.py`: Flow-based expressive model (need update)
- more modelling class to be added
- `config/default.yml`: Default configuration file

## Datasets

- [Vienna4x22]()
- [ASAP]()
- more to be added

## References

- Widmer, G., Flossmann, S., & Grachten, M. (2009). "YQX plays Chopin", AI Magazine
- [Partitura Documentation](https://partitura.readthedocs.io/)

## License

[MIT License](LICENSE)


