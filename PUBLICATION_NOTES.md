# Source release notes

This repository distributes the revised v10 Python source, output-free Colab notebook, dependency lists, and execution instructions. The Python files and notebook match the source in the archived `SER_Project_v10.zip` package. Publishing this source does not represent a new training run or new performance result.

Dataset audio, recorded predictions, trained model files, cached embeddings, individual run metadata, historical notebooks, and private notebook links are not included. Generate new experiment outputs locally using the README commands. The external emotion2vec encoder is acquired separately with the pinned downloader. Dataset and model licenses remain independent of project-code rights.

The root `SHA256SUMS.json` checks the files in this source snapshot, excluding itself. It is not a checksum inventory for the separately archived experiment artifacts.

Preparation checks parsed all published Python source files and validated the notebook JSON. The notebook contains no saved cell outputs. Existing unit tests could not start in this publication environment because `soundfile` was unavailable; no fresh dependency-installed test run, model training, or GUI test is claimed.

No project-wide license grant has been selected for the project code.
