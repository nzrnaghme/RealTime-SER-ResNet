"""Reusable CPU inference for frozen original emotion2vec base + logistic/SVM head.

``Emotion2vecPredictor(model_path, head_path).predict_file(wav)`` returns the
same ``(prediction, timing, audio_info)`` tuple as ``ser_infer.Predictor``.
The encoder is loaded once per instance. Complete-file timing is processing
latency; it excludes recording time and is not a streaming latency measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

CLASSES = ["neutral", "happy", "sad", "angry"]
MODEL_REVISION = "0c9a3152734f9d7a7a05b4ee6bfb3c109d288664"
PINNED_SHA256 = {
    "emotion2vec_base.pt": "4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd",
    "config.yaml": "844e4728d4ca9ab355a2f77b905c7a2e74fe21ed3294444e88d1e52fd874f4ef",
    "configuration.json": "bf0717f50c1e4a016902f7a24b68f3023a1fc0fb539658a6072d61d6681fc57f",
}


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


class Emotion2vecPredictor:
    """A verified, frozen encoder and the validation-selected trained head.

    ``model_path`` is the acquired model directory or its checkpoint file.
    ``head_path`` is the selected local .joblib pipeline, with ``selection.json``
    and ``config.json`` in the same directory. Only trusted project-generated
    joblib files should be passed, because joblib deserialization executes code.
    """

    def __init__(self, model_path: str | Path, head_path: str | Path,
                 threads: int = 2):
        started = time.perf_counter()
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        directory = Path(model_path).resolve()
        if directory.is_file():
            if directory.name != "emotion2vec_base.pt":
                raise ValueError("Expected emotion2vec_base.pt or its acquired directory")
            directory = directory.parent
        self.model_path = directory / "emotion2vec_base.pt"
        self.head_path = Path(head_path).resolve()
        if self.head_path.suffix != ".joblib":
            raise ValueError("Expected the selected .joblib classification pipeline")
        acquisition_path = directory / "acquisition.json"
        acquisition = read_json(acquisition_path)
        if (acquisition.get("model_id") != "emotion2vec/emotion2vec_base"
                or acquisition.get("revision") != MODEL_REVISION):
            raise ValueError("Model acquisition differs from the pinned original emotion2vec base")
        records = {entry["file"]: entry for entry in acquisition["files"]}
        actual_hashes = {}
        for name, expected in PINNED_SHA256.items():
            path = directory / name
            actual_hashes[name] = sha256(path)
            if actual_hashes[name] != expected or records.get(name, {}).get("sha256") != expected:
                raise ValueError(f"Pinned model asset mismatch: {name}")
            if path.stat().st_size != records[name]["bytes"]:
                raise ValueError(f"Acquired model asset size mismatch: {name}")
        license_path = directory / "MODEL_LICENSE"
        if sha256(license_path) != records.get("MODEL_LICENSE", {}).get("sha256"):
            raise ValueError("Model license differs from the acquisition record")
        configuration = read_json(directory / "configuration.json")
        if configuration.get("file_path_metas") != {
                "init_param": "emotion2vec_base.pt", "config": "config.yaml"}:
            raise ValueError("Configuration does not point to the verified local assets")

        config_path = self.head_path.parent / "config.json"
        selection_path = self.head_path.parent / "selection.json"
        config, selection = read_json(config_path), read_json(selection_path)
        head_hash, config_hash = sha256(self.head_path), sha256(config_path)
        if selection.get("classes") != CLASSES:
            raise ValueError("Selected head class order differs from inference")
        if head_hash != selection.get("model_sha256"):
            raise ValueError("Head differs from the validation-selected artifact")
        if config_hash != selection.get("config_sha256"):
            raise ValueError("Head configuration differs from the selection record")
        if (config.get("upstream_revision") != MODEL_REVISION
                or config.get("upstream_sha256") != PINNED_SHA256["emotion2vec_base.pt"]):
            raise ValueError("Head was not configured for this frozen upstream checkpoint")

        import joblib
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        from funasr import AutoModel
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        self.np, self.sf, self.librosa, self.torch = np, sf, librosa, torch
        self.head = joblib.load(self.head_path)
        if not isinstance(self.head, Pipeline) or len(self.head.steps) != 2:
            raise ValueError("Expected a two-step StandardScaler/classifier pipeline")
        self.scaler, self.classifier = [step for _, step in self.head.steps]
        if not isinstance(self.scaler, StandardScaler) or not isinstance(self.classifier, (LogisticRegression, SVC)):
            raise ValueError("Expected StandardScaler followed by LogisticRegression or SVC")
        self.head_kind = "logistic" if isinstance(self.classifier, LogisticRegression) else "svm"
        if self.head_kind == "svm" and (self.classifier.probability
                                       or self.classifier.kernel != "rbf"
                                       or self.classifier.decision_function_shape != "ovr"):
            raise ValueError("Expected RBF SVC with OVR decision scores and probability=False")
        if self.scaler.n_features_in_ != 768 or self.classifier.n_features_in_ != 768:
            raise ValueError("Expected 768-dimensional frozen emotion2vec embeddings")
        if not np.array_equal(self.classifier.classes_, np.arange(4)):
            raise ValueError("Expected classifier class IDs 0,1,2,3")
        if self.classifier.C != selection.get("C"):
            raise ValueError("Classifier C differs from the validation selection")

        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # An enclosing application may have initialized the shared
                # PyTorch runtime; report its actual setting instead of claiming
                # a thread setting that could not be applied.
                pass
        self.encoder = AutoModel(model=str(directory), device="cpu", ngpu=0, ncpu=threads,
                                 disable_update=True, disable_pbar=True)
        if torch.get_num_threads() != threads:
            raise RuntimeError("FunASR did not retain the requested CPU thread count")
        self.requested_threads = threads
        self.encoder.model.eval()
        for parameter in self.encoder.model.parameters():
            parameter.requires_grad_(False)
        if (self.encoder.model.training
                or any(parameter.requires_grad for parameter in self.encoder.model.parameters())):
            raise RuntimeError("Encoder was not frozen in evaluation mode")
        if not bool(self.encoder.model.cfg.normalize):
            raise ValueError("Upstream waveform normalization unexpectedly disabled")
        if self.encoder.model.proj is not None:
            raise ValueError("Expected the original embedding encoder with no pretrained classification head")

        self.kind = "emotion2vec"
        self.backend = f"Frozen original emotion2vec base (FunASR/PyTorch CPU) + saved StandardScaler/{self.head_kind} head"
        self.artifacts = {
            "model_path": str(self.model_path), "model_sha256": actual_hashes["emotion2vec_base.pt"],
            "model_revision": MODEL_REVISION,
            "model_config_sha256": actual_hashes["config.yaml"],
            "model_configuration_sha256": actual_hashes["configuration.json"],
            "acquisition_sha256": sha256(acquisition_path),
            "head_path": str(self.head_path), "head_sha256": head_hash,
            "head_kind": self.head_kind,
            "training_config_sha256": config_hash,
            "selection_sha256": sha256(selection_path),
            "inference_implementation_sha256": sha256(__file__),
            "cpu_requested_intraop_threads": threads,
            "cpu_intraop_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchaudio", "funasr", "numpy", "librosa", "soundfile", "scikit-learn")},
            "preprocessing": "float32 decode; arithmetic-mean downmix; librosa soxr_hq to 16 kHz; upstream per-utterance waveform layer_norm; final-layer temporal mean (768); saved training-fitted scaler",
        }
        self.initialization_ms = (time.perf_counter() - started) * 1000

    def predict_file(self, audio_path: str | Path):
        """Classify one already available complete audio file; never fit anything."""
        np = self.np
        started = time.perf_counter()
        native, sr = self.sf.read(audio_path, dtype="float32", always_2d=True)
        if sr <= 0 or len(native) == 0 or not np.isfinite(native).all():
            raise ValueError("Audio must have a positive rate and nonempty finite samples")
        if float(np.max(np.abs(native))) <= 1e-8:
            raise ValueError("No usable audio: input samples are effectively zero")
        decoded = time.perf_counter()
        audio = native.mean(axis=1)
        if sr != 16000:
            audio = self.librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        audio = np.ascontiguousarray(audio, dtype="float32")
        if len(audio) <= 400 or not np.isfinite(audio).all():
            raise ValueError("Audio must contain more than 400 finite samples at 16 kHz")
        if float(np.max(np.abs(audio))) <= 1e-8:
            raise ValueError("No usable audio after downmix/resampling")
        resampled = time.perf_counter()
        with self.torch.inference_mode():
            result = self.encoder.generate(input=audio, fs=16000, granularity="utterance",
                                           extract_embedding=True, disable_pbar=True)
        if self.torch.get_num_threads() != self.requested_threads:
            raise RuntimeError("Encoder inference changed the requested CPU thread count")
        embedding = np.asarray(result[0]["feats"], dtype="float32")
        if embedding.shape != (768,) or not np.isfinite(embedding).all():
            raise ValueError("Encoder returned invalid embedding dimensions or values")
        extracted = time.perf_counter()
        normalized = self.scaler.transform(embedding[None, :])
        if not np.isfinite(normalized).all():
            raise ValueError("Saved scaler produced nonfinite features")
        standardized = time.perf_counter()
        if self.head_kind == "logistic":
            scores = np.asarray(self.classifier.predict_proba(normalized))[0]
            selected_index = None
        else:
            # SVC's voting/tie semantics are not guaranteed to equal argmax of
            # its OVR decision values. Preserve the actual predicted class.
            selected_index = int(self.classifier.predict(normalized)[0])
            scores = np.asarray(self.classifier.decision_function(normalized))[0]
        inferred = time.perf_counter()
        if scores.shape != (4,) or not np.isfinite(scores).all():
            raise ValueError("Classifier returned invalid scores")
        if self.head_kind == "logistic":
            if not np.isclose(scores.sum(), 1.0):
                raise ValueError("Logistic probability scores do not sum to one")
            index = int(self.classifier.classes_[scores.argmax()])
            score_kind = "uncalibrated_logistic_probability"
            score_note = "Logistic probability scores are uncalibrated; they are not measured accuracy or calibrated confidence."
        else:
            index = selected_index
            score_kind = "svm_ovr_decision_values_not_probabilities"
            score_note = "Uncalibrated SVM decision values; do not interpret as percentages or confidence."
        prediction = {
            "label": CLASSES[index], "label_id": index, "score": float(scores[index]),
            "score_kind": score_kind,
            "scores": dict(zip(CLASSES, map(float, scores))),
            "score_interpretation": score_note,
        }
        done = time.perf_counter()
        duration = len(native) / sr
        timing = {
            "load_decode_ms": (decoded - started) * 1000,
            "resample_feature_ms": (extracted - decoded) * 1000,
            "normalization_ms": (standardized - extracted) * 1000,
            "model_call_ms": (inferred - standardized) * 1000,
            "postprocess_ms": (done - inferred) * 1000,
            "total_ready_file_ms": (done - started) * 1000,
            "processing_rtf": (done - started) / duration,
            "resample_ms": (resampled - decoded) * 1000,
            "encoder_call_ms": (extracted - resampled) * 1000,
        }
        audio_info = {"path": str(Path(audio_path).resolve()), "duration_s": duration,
                      "native_sample_rate_hz": int(sr), "native_channels": int(native.shape[1]),
                      "resampled_duration_s": len(audio) / 16000.0}
        return prediction, timing, audio_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Complete readable audio file")
    parser.add_argument("--model", required=True, help="Acquired emotion2vec base directory")
    parser.add_argument("--head", required=True, help="Validation-selected logistic or SVM .joblib file")
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    predictor = Emotion2vecPredictor(args.model, args.head, threads=args.threads)
    prediction, timing, audio_info = predictor.predict_file(args.audio)
    report = {"scope": "Full-utterance offline prediction from an already available audio file",
              "timing_scope": "Decode, downmix/resample, encoder, saved scaler, classification head and postprocessing. Excludes recording, startup, initialization, JSON serialization and UI. resample_ms and encoder_call_ms subdivide resample_feature_ms; do not add them twice.",
              "rtf_definition": "total_ready_file_seconds / original_audio_duration_seconds",
              "model_kind": predictor.kind, "artifacts": predictor.artifacts,
              "initialization_ms": predictor.initialization_ms,
              "prediction": prediction, "timing": timing, "audio": audio_info}
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(str(args.output.resolve()))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
