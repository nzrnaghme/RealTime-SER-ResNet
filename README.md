# Speech Emotion Recognition: Residual and Pretrained Models

Reproducible four-class speech emotion recognition (SER) experiments using **RAVDESS** and **TESS**. This repository contains the revised **v10** code and recorded artifacts accompanying the manuscript *Speaker Independent Speech Emotion Recognition with Residual and Pretrained Models*.

The historical repository name, `RealTime-SER-ResNet`, is retained. The current evaluation uses **complete audio files**; it does not demonstrate microphone streaming or first-prediction latency.

[فارسی](README_FA.md) · [Full results](FINAL_RESULTS.md) · [Inference guide](INFERENCE_README.md) · [Pretrained baseline guide](MODERN_BASELINE_README.md) · [CPU measurements](CPU_BENCHMARK.md)

## What is included

- Data inventory, checksums, exact speaker splits, and duplicate detection.
- Matched CNN and residual models with 101,332 learned parameters, three training seeds, and clipping/scheduler ablations.
- An acoustic SVM baseline and frozen original **emotion2vec base** with logistic-regression and RBF-SVM heads.
- Trained compact models, fitted heads, training histories, frozen selection records, and sample-level predictions under `experiments/`.
- Cached 768-dimensional emotion2vec embeddings under `precomputed/`.
- File-based inference scripts, a Tk desktop interface, and a Colab notebook for the primary CNN/residual/SVM experiment.

**Raw audio and the large emotion2vec encoder checkpoint are not included.** Dataset/model acquisition is described below. The compact Mel/acoustic feature cache is also not included here; generate it from the official audio.

## Recorded results and limits

The validation-selected system is **emotion2vec + RBF SVM (C = 1)**. Scores below are percentages on the same fixed speaker-disjoint test partition.

| System | Test accuracy | Pooled UAR | Macro-F1 | Corpus-balanced UAR |
|---|---:|---:|---:|---:|
| Acoustic SVM | 69.30 | 69.58 | 63.89 | 66.95 |
| emotion2vec + logistic regression | 63.49 | 63.63 | 61.63 | 71.52 |
| emotion2vec + RBF SVM | 67.87 | 68.17 | 64.42 | 72.50 |

Corpus-balanced UAR is the equally weighted mean of RAVDESS and TESS UAR. The selected system gains **5.55 percentage points** on this metric relative to the acoustic SVM, while pooled accuracy falls by **1.43 points**. Residual connections and stabilization controls show no consistent advantage in the matched compact-model comparison. All neural configurations and negative results are retained in [FINAL_RESULTS.md](FINAL_RESULTS.md).

A substantial unresolved failure is **3.50% TESS Angry recall (7/200)**: 189 of those 200 recordings are classified as Happy. Results concern acted English speech, four labels, and one fixed partition. They do not establish natural-emotion recognition, Persian-language performance, clinical validity, or leave-one-corpus-out transfer. The older paper's 74.4% test accuracy and 80.36% peak validation accuracy used a different, unclear evaluation protocol and are **not directly comparable**.

## Data and speaker partitions

There are **2,271 unique recordings**: 671 RAVDESS and 1,600 TESS. One exact RAVDESS training duplicate is removed and documented. Label order throughout the project is `neutral`, `happy`, `sad`, `angry`.

| Partition | RAVDESS speakers | RAVDESS clips | TESS speaker | TESS clips | Total |
|---|---|---:|---|---:|---:|
| Training | Actors 01–16 | 447 | YAF | 800 | 1,247 |
| Validation | Actors 17–20 | 112 | None | 0 | 112 |
| Test | Actors 21–24 | 112 | OAF | 800 | 912 |

Speakers do not overlap across partitions. Lexical material does overlap. TESS has only two speakers, so this three-way speaker split has no TESS validation speaker. Seed variability does not measure uncertainty over a population of unseen speakers.

Get the official audio from:

- [RAVDESS audio-only speech archive](https://zenodo.org/records/1188976)
- [Toronto emotional speech set (TESS)](https://doi.org/10.5683/SP2/E8H2MF)

Respect the source licenses. Use the original recordings, not previously augmented or merged datasets. `SER_v10_Colab.ipynb` includes download and source-checksum verification steps. Canonical IDs and the recorded split are in `data_inventory/`; `acquisition.json` records acquisition provenance. Existing manifests preserve paths from the recorded run. Rebuild the manifest for fresh audio extraction on another machine; do not edit a frozen manifest in place.

## Set up the compact-model environment

The recorded environment uses **Python 3.12**. Use separate virtual environments for the compact pipeline and emotion2vec because their pinned dependencies differ. The following commands use a Bash shell and run from the repository root.

```bash
git clone https://github.com/nzrnaghme/RealTime-SER-ResNet.git
cd RealTime-SER-ResNet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, use the Python 3.12 installation and `.venv\Scripts\Activate.ps1` in PowerShell instead of `source`. The Tk interface needs Tkinter and a graphical desktop; terminal inference does not.

## Reproduce a new compact-model experiment

Place the unpacked official recordings beneath `data/RAVDESS` and `data/TESS`. Keep existing artifacts unchanged and use **fresh output directories**.

```bash
python ser_data_v10.py \
  --ravdess data/RAVDESS --tess data/TESS \
  --out runs/new_features --workers 2

python ser_models_v10.py train \
  --features runs/new_features/features.npz \
  --manifest runs/new_features/manifest.csv \
  --output runs/new_foldA \
  --models svm cnn r0 r1 r2 r3 --seeds 17 42 73 --epochs 25
```

The feature extractor resamples to 16 kHz and produces a 128 × 128 log-Mel representation of each complete utterance, plus 252 acoustic descriptors. Training fits normalization using training data only. The primary model labels are:

| Label | Configuration |
|---|---|
| `svm` | Acoustic RBF SVM |
| `cnn` | Matched CNN without residual additions |
| `r0` | Residual model without clipping/scheduling |
| `r1` | Residual model with clipping only |
| `r2` | Residual model with scheduling only |
| `r3` | Residual model with clipping and scheduling |

Review validation results and finish all configuration choices before freezing. Then evaluate the held-out test set:

```bash
python ser_models_v10.py freeze --output runs/new_foldA
python ser_models_v10.py evaluate \
  --features runs/new_features/features.npz \
  --manifest runs/new_features/manifest.csv \
  --output runs/new_foldA
```

A frozen campaign cannot be retrained in place. Further model development after inspecting test results requires a new independent evaluation. The supplementary masking/dropout experiment uses `ser_models_regularized_v10.py`; its rationale and recorded configuration are in `REGULARIZED_CANDIDATE_PRECOMMIT.md` and `experiments/foldA_regularized/`.

### Colab

Open [SER_v10_Colab.ipynb](SER_v10_Colab.ipynb) in Colab with a fresh Python 3.12 runtime and run the cells in order. It embeds the primary pipeline source and downloads the official corpora; Drive access is not required. Test evaluation is disabled initially through `RUN_HELD_OUT = False`. Enable it only after reviewing validation and closing model choices. The notebook does not run the supplementary regularized or emotion2vec experiments; their scripts and guides are separate. Download your outputs before the temporary Colab runtime expires.

## Predict from a complete audio file

Using the included acoustic SVM in the compact-model environment:

```bash
python ser_infer.py recording.wav \
  --model experiments/foldA/svm/best.joblib \
  --feature-metadata data_inventory/feature_metadata.json \
  --output prediction.json

python ser_demo.py \
  --model experiments/foldA/svm/best.joblib \
  --metadata data_inventory/feature_metadata.json
```

For a compact neural model, use its `best.keras` checkpoint and preserve the neighboring `normalizer.npz`, `config.json`, and `result.json`. SVM decision values are not probabilities. Softmax and logistic outputs are uncalibrated; none of these scores is measured accuracy.

## Use or reproduce the emotion2vec baseline

Create a separate environment. The pinned encoder downloader verifies the exact revision and SHA-256 values; its external checkpoint is approximately **1.126 GB**.

```bash
python3.12 -m venv .venv-emotion2vec
source .venv-emotion2vec/bin/activate
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_emotion2vec.txt
python -m pip check
```

**Re-evaluate saved heads without audio or encoder downloads:**

```bash
python evaluate_emotion2vec_v10.py \
  --frozen GLOBAL_FROZEN_SELECTIONS.json \
  --manifest data_inventory/manifest.csv \
  --features precomputed/emotion2vec_features.npz \
  --head-root experiments \
  --out runs/reproduced_emotion2vec_test
```

This verifies the frozen artifacts and performs no fitting. Use a new output directory on each run. The relocation option `--head-root experiments` is required when the repository is not at the original experiment path.

**Run the selected model on a new file:**

```bash
python download_emotion2vec.py --output emotion2vec_base
python emotion2vec_predictor_v10.py recording.wav \
  --model emotion2vec_base \
  --head experiments/emotion2vec_svm_head/C1.joblib \
  --threads 2 --output prediction.json

python ser_demo.py \
  --encoder emotion2vec_base \
  --model experiments/emotion2vec_svm_head/C1.joblib
```

Keep the exact original `emotion2vec_base` checkpoint; substituting emotion2vec+ does not reproduce this experiment. For extraction and refitting in fresh directories, follow [MODERN_BASELINE_README.md](MODERN_BASELINE_README.md). Upstream pretraining was not independently audited at the recording level.

## Checks

Run the existing synthetic data/inference checks in the compact-model environment:

```bash
python -m unittest -v test_ser_data_v10 test_ser_infer
```

These tests check parsing, split rules, duplicate handling, and inference consistency. They are not SER performance experiments. Source acquisition, encoder assets, and predictor verification also have dedicated scripts: `verify_acquisition.py`, `verify_emotion2vec_assets.py`, and `verify_emotion2vec_predictor.py`; inspect their arguments before using them with your local data paths.

## Runtime interpretation

The recorded CPU benchmark uses 40 complete recordings, three measured repetitions per model, and three warmups. Median ready-file processing was **13.34 ms** for the acoustic SVM, **15.80 ms** for the regularized residual model (seed 17), and **346.37 ms** for emotion2vec + SVM. These measurements exclude recording, model initialization, and UI overhead. RTF below one does not establish streaming latency. Hardware, thread settings, and all measurements are preserved in `CPU_BENCHMARK.json`.

The desktop application is file-based. Its inference paths were checked against saved models; the archived v10 release does not claim a completed visual test of the Tk interface.

## Provenance and third-party material

`GLOBAL_FROZEN_SELECTIONS.json`, campaign freeze files, and per-run configurations preserve the recorded experiment artifacts. `SHA256SUMS.json` verifies this GitHub snapshot; the unchanged checksum list for the original v10 archive is retained as `provenance/archive_SHA256SUMS.json`. Historical notebooks, private notebook links, the annotated old manuscript, and an orphan temporary model file are excluded from this public repository and remain in the original archive. See [Publication notes](PUBLICATION_NOTES.md).

Dataset and pretrained-model licenses are independent of the project code. The included emotion2vec attribution and model license remain under `emotion2vec_base/`; this README does not grant additional rights to third-party material. The accompanying manuscript is a research manuscript, not a claim of an accepted publication.

No project-wide open-source license has been selected for the project code. Third-party licenses apply only to their respective material.
