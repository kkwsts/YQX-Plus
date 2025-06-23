# YQX+: Benchmarking Probabilistic Modelling of Performance Expression

This repository expand on the idea of the [YQX system]() for expressive music performance, that 
The system predicts expressive performance parameters from musical score context using a Bayesian model trained on human performances.


## Overview

- Extracts musical context features from MusicXML scores
- Trains a probabilitis expressive model on aligned score-performance data
- Predicts expressive parameters for new scores
- Renders expressive MIDI performances

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

### Training the Model

```bash
python yqx.py --train  [--use_asap --asap_dir <path_to_asap>]
```

- `--train`: Train the model
- `--use_asap`: (Optional) Use ASAP dataset for training
- `--asap_dir`: (Optional) Path to ASAP dataset directory
- `--use_midihum`: (Optional) Use MIDIHUM features

### Rendering an Expressive Performance

```bash
python yqx.py --render --input_score <path_to_musicxml> --output_midi <output_midi_path> --model_path <model_file>
```

- `--input_score`: Path to input MusicXML score
- `--output_midi`: Path for output MIDI file
- `--model_path`: Path to trained model file (default: yqx_model.pkl)

### Features 

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

## Datasets

- [Vienna4x22]()
- [ASAP]()
- more to be added

## References

- Widmer, G., Flossmann, S., & Grachten, M. (2009). "YQX plays Chopin", AI Magazine
- [Partitura Documentation](https://partitura.readthedocs.io/)

## License

[MIT License](LICENSE)


