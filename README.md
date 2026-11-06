# Speech Emotion Recognition: Residual and Pretrained Models

Source code for the revised **v10** four-class speech emotion recognition (SER) study using RAVDESS and TESS. Classes are `neutral`, `happy`, `sad`, and `angry`.

The historical repository name, `RealTime-SER-ResNet`, is retained. The current inference pipeline processes **complete audio files**; it does not establish microphone streaming or first-prediction latency.

[فارسی](README_FA.md) · [Source release notes](PUBLICATION_NOTES.md)

## Source release

This repository contains the Python training, extraction, inference, verification and test scripts, an output-free Colab notebook, and pinned dependency files. **It does not contain trained weights, feature caches, audio, saved predictions, experiment manifests, or archived run metadata.** Running the commands below creates a new local experiment; cloning this repository alone does not reproduce the original archived run.

The study compares matched CNN/residual models, clipping/scheduling ablations, an acoustic SVM, and frozen original emotion2vec representations with logistic-regression and RBF-SVM heads. The previously recorded validation-selected emotion2vec + SVM system achieved **67.87% test accuracy, 68.17% UAR, 64.42% macro-F1, and 72.50% corpus-balanced UAR**. These are recorded research results, not outcomes produced by downloading the source. They do not establish an improvement over the older paper's 74.4% accuracy, which used a different evaluation protocol.

## Data

Download the official [RAVDESS audio-only speech archive](https://zenodo.org/records/1188976) and [TESS dataset](https://doi.org/10.5683/SP2/E8H2MF), respecting their licenses. Extract them beneath `data/RAVDESS` and `data/TESS`. Do not substitute previously augmented or merged data.

The inventory code defines fixed speaker-disjoint partitions, detects duplicates, and writes local manifests and checksums. TESS has only two speakers and is absent from validation in this protocol. Lexical content is not held out. These acted English recordings do not establish performance on natural emotions, other languages, or clinical applications.

## Compact models: installation and a fresh experiment

Use Python 3.12 and run the following Bash commands from the repository root. The compact and emotion2vec pipelines need separate environments because their pinned dependencies differ.

```bash
git clone https://github.com/nzrnaghme/RealTime-SER-ResNet.git
cd RealTime-SER-ResNet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

python ser_data_v10.py \
  --ravdess data/RAVDESS --tess data/TESS \
  --out runs/new_features --workers 2

python ser_models_v10.py train \
  --features runs/new_features/features.npz \
  --manifest runs/new_features/manifest.csv \
  --output runs/new_foldA \
  --models svm cnn r0 r1 r2 r3 --seeds 17 42 73 --epochs 25
```

On Windows, activate the environment using `.venv\Scripts\Activate.ps1` in PowerShell. `cnn` is the matched non-residual model; `r0` has no clipping/scheduler, `r1` uses clipping, `r2` uses scheduling, and `r3` uses both. The supplementary masking/dropout implementation is in `ser_models_regularized_v10.py`.

Review validation results and finish model choices **before** freezing and evaluating the held-out test partition:

```bash
python ser_models_v10.py freeze --output runs/new_foldA
python ser_models_v10.py evaluate \
  --features runs/new_features/features.npz \
  --manifest runs/new_features/manifest.csv \
  --output runs/new_foldA
```

Use fresh output directories for new experiments. Frozen campaigns must not be retrained in place. Additional tuning after seeing test results needs an independent evaluation.

## Predict with your newly trained acoustic SVM

After training above, provide a complete readable audio file:

```bash
python ser_infer.py recording.wav \
  --model runs/new_foldA/svm/best.joblib \
  --feature-metadata runs/new_features/feature_metadata.json \
  --output prediction.json

python ser_demo.py \
  --model runs/new_foldA/svm/best.joblib \
  --metadata runs/new_features/feature_metadata.json
```

The desktop interface requires Tkinter and a graphical desktop. Neural inference uses a generated `best.keras` checkpoint with its neighboring normalizer and configuration files. SVM decision scores are not probabilities; softmax/logistic scores are not calibrated confidence or measured accuracy.

## emotion2vec: acquire the encoder, extract, and fit

The source includes a downloader for the exact original emotion2vec base checkpoint, with revision and checksum verification. The external checkpoint is approximately 1.126 GB. Dataset and model licenses remain independent of this project.

```bash
python3.12 -m venv .venv-emotion2vec
source .venv-emotion2vec/bin/activate
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_emotion2vec.txt
python -m pip check

python download_emotion2vec.py --output emotion2vec_base
python ser_data_v10.py \
  --ravdess data/RAVDESS --tess data/TESS \
  --out runs/emotion2vec_inventory --inventory-only
python emotion2vec_baseline_v10.py extract \
  --model emotion2vec_base \
  --manifest runs/emotion2vec_inventory/manifest.csv \
  --out runs/new_emotion2vec_features.npz --threads 2
python emotion2vec_baseline_v10.py fit \
  --manifest runs/emotion2vec_inventory/manifest.csv \
  --features runs/new_emotion2vec_features.npz \
  --out runs/new_logistic_head
python emotion2vec_svm_v10.py \
  --manifest runs/emotion2vec_inventory/manifest.csv \
  --features runs/new_emotion2vec_features.npz \
  --out runs/new_svm_head
```

Both fitting scripts use training data for fitting and validation UAR for choosing among their bounded candidate sets. They write a local `selection.json` identifying the chosen model. `emotion2vec_predictor_v10.py` provides file inference through `--model` (the encoder directory) and `--head` (your selected `.joblib` file).

`evaluate_emotion2vec_v10.py` is included as source, but requires an explicit global freeze record and its matching artifacts. Those original records are not published here. Do not treat a newly fitted model as the archived experiment or claim exact replay without the matching records. The included benchmark/verification scripts likewise require the local inputs named in their arguments.

## Colab and checks

[SER_v10_Colab.ipynb](SER_v10_Colab.ipynb) contains the primary CNN/residual/acoustic-SVM workflow, including official-data downloads and checksum checks. Use a fresh Python 3.12 runtime and run cells in order. It starts with test evaluation disabled (`RUN_HELD_OUT = False`); enable it only after closing validation-based choices. The notebook does not run the separate regularized or emotion2vec experiments. Save generated outputs before the temporary runtime expires.

Run the existing synthetic checks in the compact environment:

```bash
python -m unittest -v test_ser_data_v10 test_ser_infer
```

These tests verify parsing, duplicate handling, split rules, and inference consistency. They are not SER performance experiments. No raw data, generated models, or test results are committed by these commands; keep generated files local unless you separately choose to release them.

No project-wide license grant has been selected for the project code. Dataset and model licenses apply separately.
