#!/usr/bin/env python3
"""Full-utterance SER inference and auditable ready-file timing for v10 models.

Uses the same feature implementation and saved training normalizers as the
experiment. Input must be an already available complete audio file. This is not
a microphone or streaming latency measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for name, value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "2",
                    "TF_NUM_INTEROP_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "2"}.items():
    os.environ.setdefault(name, value)

# Both a flat released code bundle and the development tree are supported.
if not (Path(__file__).resolve().parent / "ser_data_v10.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_review"))

import joblib
import numpy as np
import soundfile as sf
import ser_data_v10 as features


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


class Predictor:
    def __init__(self, model_path: str | Path, feature_metadata: str | Path,
                 normalizer_path: str | Path | None = None,
                 frozen_selections: str | Path | None = None,
                 compiled: bool = True):
        started = time.perf_counter()
        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(feature_metadata).resolve()
        metadata = read_json(self.metadata_path)
        if metadata.get("feature_configuration_sha256") != features.FEATURE_CONFIG_HASH:
            raise ValueError("Model feature metadata does not match the shared feature implementation")
        self.artifacts = {"model_path": str(self.model_path),
                          "model_sha256": sha256(self.model_path),
                          "feature_metadata_path": str(self.metadata_path),
                          "feature_metadata_sha256": sha256(self.metadata_path),
                          "feature_configuration_sha256": features.FEATURE_CONFIG_HASH,
                          "feature_implementation_sha256": sha256(features.__file__)}
        config_path = self.model_path.parent / "config.json"
        if config_path.exists():
            config = read_json(config_path)
            if config.get("labels") != features.CLASSES:
                raise ValueError("Training label order differs from inference label order")
            if config.get("features_sha256") != metadata.get("features_sha256"):
                raise ValueError("Feature metadata does not identify the training feature cache")
            self.artifacts["training_config_sha256"] = sha256(config_path)
        result_path = self.model_path.parent / "result.json"
        result = read_json(result_path) if result_path.exists() else {}
        if result and result.get("checkpoint_sha256") != self.artifacts["model_sha256"]:
            raise ValueError("Checkpoint hash differs from the completed training result")

        if self.model_path.suffix == ".joblib":
            self.kind = "svm"
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            self.model = joblib.load(self.model_path)
            if not isinstance(self.model, Pipeline) or len(self.model.steps) != 2:
                raise ValueError("Expected the v10 StandardScaler/SVC pipeline")
            self.scaler, self.classifier = [step for _, step in self.model.steps]
            if not isinstance(self.scaler, StandardScaler) or not isinstance(self.classifier, SVC):
                raise ValueError("Expected StandardScaler followed by SVC")
            if self.scaler.n_features_in_ != 252 or self.classifier.probability:
                raise ValueError("Expected 252 acoustic features and uncalibrated SVC")
            if not np.array_equal(self.classifier.classes_, np.arange(4)):
                raise ValueError("Expected SVM class IDs 0,1,2,3")
            self.backend = "scikit-learn saved StandardScaler + SVC"
        elif self.model_path.suffix == ".keras":
            self.kind = "neural"
            import tensorflow as tf
            self.tf = tf
            try:
                tf.config.threading.set_intra_op_parallelism_threads(2)
                tf.config.threading.set_inter_op_parallelism_threads(2)
            except RuntimeError:
                pass
            normalizer_path = Path(normalizer_path or self.model_path.parent / "normalizer.npz").resolve()
            normalizer_hash = sha256(normalizer_path)
            if result and result.get("normalizer_sha256") != normalizer_hash:
                raise ValueError("Normalizer hash differs from the completed training result")
            with np.load(normalizer_path, allow_pickle=False) as archive:
                self.mean, self.std = archive["mean"], archive["std"]
            if self.mean.shape != (1, 128, 1, 1) or self.std.shape != self.mean.shape:
                raise ValueError("Expected saved per-Mel-band normalizer shape (1,128,1,1)")
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
                raise ValueError("Normalizer contains invalid values")
            self.artifacts.update(normalizer_path=str(normalizer_path), normalizer_sha256=normalizer_hash)
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            if self.model.input_shape != (None, 128, 128, 1) or self.model.output_shape != (None, 4):
                raise ValueError("Expected v10 log-Mel input and four logits")
            if compiled:
                @tf.function(input_signature=[tf.TensorSpec((None, 128, 128, 1), tf.float32)])
                def infer(x):
                    return self.model(x, training=False)
                self.infer = infer
                self.backend = "TensorFlow tf.function; training=False; output synchronized by .numpy()"
            else:
                self.infer = lambda x: self.model(x, training=False)
                self.backend = "TensorFlow eager; training=False; output synchronized by .numpy()"
        else:
            raise ValueError("Model must be a v10 .joblib pipeline or .keras checkpoint")

        frozen_path = Path(frozen_selections) if frozen_selections else self.model_path.parent.parent / "FROZEN_SELECTIONS.json"
        if frozen_path.exists():
            frozen = read_json(frozen_path)
            matches = [entry for entry in frozen["runs"] if entry["run"] == self.model_path.parent.name]
            if len(matches) != 1:
                raise ValueError("Model run is absent from the frozen campaign")
            entry = matches[0]
            if entry["checkpoint_sha256"] != self.artifacts["model_sha256"]:
                raise ValueError("Model differs from the frozen checkpoint")
            if entry.get("normalizer_sha256") != self.artifacts.get("normalizer_sha256"):
                raise ValueError("Normalizer differs from the frozen checkpoint")
            if entry["config_sha256"] != self.artifacts.get("training_config_sha256"):
                raise ValueError("Training configuration differs from the frozen campaign")
            self.artifacts.update(frozen_selections_path=str(frozen_path.resolve()),
                                  frozen_selections_sha256=sha256(frozen_path),
                                  validation_selected_configuration=frozen["validation_selected_configuration"])
        elif frozen_selections:
            raise FileNotFoundError(frozen_path)
        self.initialization_ms = (time.perf_counter() - started) * 1000

    def normalize(self, logmel, acoustic):
        if self.kind == "svm":
            return self.scaler.transform(acoustic[None, :])
        return ((logmel[None, ...] - self.mean) / self.std).astype(np.float32)

    def model_call(self, normalized):
        if not np.isfinite(normalized).all():
            raise ValueError("Nonfinite normalized features")
        if self.kind == "svm":
            # Calling both preserves SVC's vote/tie rule: argmax(OVR scores) is
            # not a guaranteed substitute for SVC.predict with break_ties=False.
            return self.classifier.predict(normalized), self.classifier.decision_function(normalized)
        return np.asarray(self.infer(normalized).numpy())

    def postprocess(self, raw):
        if self.kind == "svm":
            pred, scores = raw
            index = int(pred[0])
            values = np.asarray(scores)[0]
            score_kind = "svm_ovr_decision_values_not_probabilities"
            note = "Uncalibrated SVM decision values; do not interpret as percentages or confidence."
        else:
            values = softmax(np.asarray(raw))[0]
            index = int(values.argmax())
            score_kind = "uncalibrated_softmax"
            note = "Softmax scores are uncalibrated; they are not measured accuracy or calibrated confidence."
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite model scores")
        return {"label": features.CLASSES[index], "label_id": index,
                "score_kind": score_kind,
                "scores": dict(zip(features.CLASSES, map(float, values))), "score_interpretation": note}

    def predict_file(self, audio_path: str | Path):
        start = time.perf_counter()
        native, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        if len(native) == 0 or not np.isfinite(native).all():
            raise ValueError("Audio must be nonempty and finite")
        if np.max(np.abs(native)) <= 1e-8:
            raise ValueError("The recording contains no usable audio signal")
        decoded = time.perf_counter()
        audio = features.resample_audio(native, sr)
        logmel, acoustic = features.features_for_audio(audio)
        extracted = time.perf_counter()
        normalized = self.normalize(logmel, acoustic)
        standardized = time.perf_counter()
        raw = self.model_call(normalized)
        inferred = time.perf_counter()
        prediction = self.postprocess(raw)
        done = time.perf_counter()
        duration = len(native) / sr
        timing = {"load_decode_ms": (decoded - start) * 1000,
                  "resample_feature_ms": (extracted - decoded) * 1000,
                  "normalization_ms": (standardized - extracted) * 1000,
                  "model_call_ms": (inferred - standardized) * 1000,
                  "postprocess_ms": (done - inferred) * 1000,
                  "total_ready_file_ms": (done - start) * 1000,
                  "processing_rtf": (done - start) / duration}
        info = {"path": str(Path(audio_path).resolve()), "duration_s": duration,
                "native_sample_rate_hz": int(sr), "native_channels": int(native.shape[1]),
                "resampled_duration_s": len(audio) / 16000.0}
        return prediction, timing, info


def environment(predictor: Predictor) -> dict:
    import librosa
    import sklearn
    import threadpoolctl
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "logical_cpu_count": os.cpu_count(), "processor": platform.processor(),
              "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "load_average_1_5_15_minutes": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
              "versions": {"numpy": np.__version__, "librosa": librosa.__version__,
                           "soundfile": sf.__version__, "scikit_learn": sklearn.__version__},
              "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
              "thread_pools": threadpoolctl.threadpool_info(), "inference_backend": predictor.backend,
              "cpu_isolation": "No CPU isolation is enforced; concurrent training or other processes can affect timing."}
    if predictor.kind == "neural":
        result["versions"]["tensorflow"] = predictor.tf.__version__
        result["tensorflow_visible_devices"] = [device.device_type for device in predictor.tf.config.get_visible_devices()]
    return result


def summarize(timings: list[dict]) -> dict:
    return {name: {"p50": float(np.percentile([row[name] for row in timings], 50)),
                   "p95": float(np.percentile([row[name] for row in timings], 95)),
                   "mean": float(np.mean([row[name] for row in timings])),
                   "minimum": float(min(row[name] for row in timings)),
                   "maximum": float(max(row[name] for row in timings))}
            for name in timings[0]}


def run(args) -> dict:
    predictor = Predictor(args.model, args.feature_metadata, args.normalizer,
                          args.frozen_selections, compiled=not args.eager)
    report = {"scope": "Full-utterance offline prediction from an already available audio file",
              "timing_scope": "Ready-file wall time includes decode, downmix/resample, shared feature extraction, training-fitted normalization, model calls, and score postprocessing. Excludes recording/waiting for speech, process startup, module imports, model initialization, JSON serialization and UI.",
              "feature_scope": "Shared extractor calculates both log-Mel and acoustic features on every pass, including the unused representation; no cache of features is used.",
              "rtf_definition": "total_ready_file_seconds / original_audio_duration_seconds; this is processing RTF, not first-prediction or microphone latency",
              "model_kind": predictor.kind, "labels": features.CLASSES,
              "artifacts": predictor.artifacts, "initialization_ms": predictor.initialization_ms,
              "initialization_scope": "Artifact checks, model/normalizer loading and backend initialization; excludes script/module startup. tf.function tracing occurs on the first prediction, if compiled.",
              "runtime_before": environment(predictor), "workload_note": args.workload_note,
              "benchmark": bool(args.benchmark)}
    if args.benchmark:
        warmups, measured = [], []
        for _ in range(args.warmups):
            _, timing, _ = predictor.predict_file(args.audio)
            warmups.append(timing)
        for _ in range(args.repeats):
            prediction, timing, audio_info = predictor.predict_file(args.audio)
            measured.append(timing)
        report.update(warmup_count=args.warmups, measured_repetitions=args.repeats,
                      warmup_raw=warmups, measured_raw=measured, measured_summary=summarize(measured),
                      summary_scope="Percentiles describe repeated processing of the same file in this process, with OS file caches and warmed libraries; they are not a distribution over speakers or audio files.",
                      prediction=prediction, audio=audio_info)
    else:
        prediction, timing, audio_info = predictor.predict_file(args.audio)
        report.update(prediction=prediction, timing=timing, audio=audio_info)
    report["runtime_after"] = environment(predictor)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("audio", help="Complete readable audio file; never treated as a streaming window")
    result.add_argument("--model", required=True, help="best.joblib or best.keras")
    result.add_argument("--feature-metadata", required=True, help="feature_metadata.json from the training feature cache")
    result.add_argument("--normalizer", help="Neural normalizer.npz; defaults beside the checkpoint")
    result.add_argument("--frozen-selections", help="Optional explicit FROZEN_SELECTIONS.json; auto-detected beside the campaign")
    result.add_argument("--eager", action="store_true", help="Use eager neural inference instead of tf.function")
    result.add_argument("--benchmark", action="store_true")
    result.add_argument("--warmups", type=int, default=3)
    result.add_argument("--repeats", type=int, default=20)
    result.add_argument("--workload-note", default="Other concurrent workload was not independently controlled.")
    result.add_argument("--output", help="JSON destination; stdout when omitted")
    return result


def main():
    args = parser().parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise SystemExit("warmups must be nonnegative and repeats must be positive")
    report = run(args)
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized)
        print(str(destination.resolve()))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
