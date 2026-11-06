#!/usr/bin/env python3
"""Prespecified full-file CPU timing; run each backend in its own Python environment.

prepare -> run-compact (SVM and neural) -> run-emotion2vec -> combine
Only the prepare command is used before project training/extraction has stopped.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

for key, value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "2",
                   "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2"}.items():
    os.environ.setdefault(key, value)

import numpy as np

ROOT = Path(__file__).resolve().parent
CLASSES = ["neutral", "happy", "sad", "angry"]
PHASES = ["load_decode_ms", "resample_feature_ms", "normalization_ms", "model_call_ms",
          "postprocess_ms", "total_ready_file_ms", "processing_rtf"]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(destination)


def prepare(manifest, output, seed=20260914):
    with Path(manifest).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for corpus, split in [("RAVDESS", "val"), ("TESS", "train")]:
        for label in CLASSES:
            eligible = sorted((row for row in rows if row["corpus"] == corpus and row["split"] == split
                               and row["label"] == label), key=lambda row: row["sample_id"])
            if len(eligible) < 5:
                raise ValueError(f"Need at least five eligible {corpus}/{split}/{label} clips")
            selected.extend(eligible[:5])
    if len(selected) != 40 or len({row["sample_id"] for row in selected}) != 40:
        raise ValueError("Benchmark must have exactly forty unique clips")
    rng = random.Random(seed)
    schedule = []
    for repetition in range(3):
        order = list(range(40))
        rng.shuffle(order)
        for index in order:
            schedule.append({"call_index": len(schedule), "repetition": repetition,
                             "sample_id": selected[index]["sample_id"]})
    plan = {"schema": "ser_cpu_benchmark_plan_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(Path(manifest).resolve()), "manifest_sha256": sha256(manifest),
            "selection_rule": "Within each class, first five sample_id-sorted RAVDESS validation clips and first five TESS training clips; no held-out test clips.",
            "selected_count": 40, "clips_per_corpus": 20, "clips_per_corpus_class": 5,
            "repetitions_per_clip": 3, "measured_calls_per_model": 120,
            "shuffle_algorithm": "Python random.Random(seed), three successive shuffles of indices 0..39; exact order retained below and shared by both backends",
            "shuffle_seed": seed, "warmup_count": 3,
            "warmup_sample_id": selected[0]["sample_id"],
            "canonical_presentation_sample_id": selected[0]["sample_id"],
            "clips": selected, "schedule": schedule,
            "execution_requirement": "Run only after parent confirms all project training/extraction stopped. Models execute sequentially in separate processes; no hardware isolation is claimed."}
    write_json(output, plan)
    return plan


def summarize(rows):
    result = {}
    for key in PHASES + ["resample_ms", "encoder_call_ms"]:
        values = [row["timing"][key] for row in rows if key in row["timing"]]
        if not values:
            continue
        result[key] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                       "mean": float(np.mean(values)), "minimum": float(min(values)), "maximum": float(max(values))}
    return {"measured_calls": len(rows), "unique_clips": len({row["sample_id"] for row in rows}),
            "statistics": result}


def grouped_summary(rows):
    groups = {"pooled": rows}
    groups.update({corpus: [row for row in rows if row["corpus"] == corpus] for corpus in ("RAVDESS", "TESS")})
    return {name: summarize(group) for name, group in groups.items() if group}


def runtime(predictor):
    import threadpoolctl
    versions = {}
    for package in ["numpy", "librosa", "soundfile", "scikit-learn", "tensorflow", "torch", "funasr", "soxr"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    cpu_model = ""
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    result = {"at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": platform.python_version(), "python_executable": sys.executable,
              "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(),
              "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "load_average_1_5_15_minutes": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
              "thread_environment": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]},
              "thread_pools": threadpoolctl.threadpool_info(), "versions": versions,
              "backend": predictor.backend, "hardware_isolation": False}
    if hasattr(predictor, "torch"):
        result["torch_threads"] = {"intraop": predictor.torch.get_num_threads(), "interop": predictor.torch.get_num_interop_threads()}
    if hasattr(predictor, "tf"):
        result["tensorflow_threads"] = {"intraop": predictor.tf.config.threading.get_intra_op_parallelism_threads(),
                                        "interop": predictor.tf.config.threading.get_inter_op_parallelism_threads()}
        result["tensorflow_visible_devices"] = [d.device_type for d in predictor.tf.config.get_visible_devices()]
    return result


def run_backend(args):
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    if sha256(plan["manifest_path"]) != plan["manifest_sha256"]:
        raise ValueError("Manifest changed after timing clips were selected")
    if len(plan["clips"]) != 40 or len(plan["schedule"]) != 120 or plan["warmup_count"] != 3:
        raise ValueError("Benchmark plan differs from the required sample/repetition counts")
    clips = {row["sample_id"]: row for row in plan["clips"]}
    if any(row["split"] != {"RAVDESS": "val", "TESS": "train"}[row["corpus"]] for row in clips.values()):
        raise ValueError("Benchmark includes an unauthorized partition")
    for row in clips.values():
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Audio file differs from the manifest: {row['sample_id']}")
    if args.command == "run-compact":
        from ser_infer import Predictor
        predictor = Predictor(args.model, args.feature_metadata, args.normalizer,
                              args.frozen_selections, compiled=True)
        backend_key = args.backend_key
    else:
        module_dir = ROOT if (ROOT / "emotion2vec_predictor_v10.py").exists() else ROOT.parent / "code_review"
        sys.path.insert(0, str(module_dir))
        from emotion2vec_predictor_v10 import Emotion2vecPredictor
        predictor = Emotion2vecPredictor(args.encoder, args.head, threads=2)
        backend_key = "emotion2vec"
    report = {"schema": "ser_cpu_benchmark_model_v1", "backend_key": backend_key,
              "model_kind": predictor.kind, "model_label": args.model_label,
              "plan_path": str(plan_path.resolve()), "plan_sha256": sha256(plan_path),
              "harness_sha256": sha256(__file__), "artifacts": predictor.artifacts,
              "initialization_ms": predictor.initialization_ms,
              "runtime_before": runtime(predictor), "workload_note": args.workload_note,
              "timing_scope": "Complete already-available file: load/decode, downmix/resample and features or frozen encoder, saved normalization, classifier, score postprocessing. Excludes audio acquisition, process/module startup, model initialization, JSON output and UI.",
              "rtf_definition": "total_ready_file_seconds / original_audio_duration_seconds",
              "latency_limit": "Processing RTF and ready-file time do not measure microphone-to-first-prediction latency; the full utterance must already be available.",
              "feature_scope": "Compact: shared extractor computes both log-Mel and acoustic features each pass. emotion2vec: resample_feature_ms includes its frozen encoder; encoder_call_ms/resample_ms are subdivisions, not additional phases.",
              "cache_scope": "All input hashes are checked before timing, so OS file caches may already be warm. Three warmups precede the repeated measurements; libraries are warm. No features or encoder embeddings are cached between measured calls.",
              "warmup_raw": [], "measured_raw": []}
    warmup_clip = clips[plan["warmup_sample_id"]]
    for index in range(3):
        prediction, timing, audio = predictor.predict_file(warmup_clip["path"])
        report["warmup_raw"].append({"warmup_index": index, "sample_id": warmup_clip["sample_id"],
                                     "corpus": warmup_clip["corpus"], "prediction": prediction,
                                     "timing": timing, "audio": audio})
    started = time.perf_counter()
    for call in plan["schedule"]:
        clip = clips[call["sample_id"]]
        prediction, timing, audio = predictor.predict_file(clip["path"])
        if not np.isclose(audio["duration_s"], float(clip["duration_s"]), rtol=0, atol=1e-10):
            raise ValueError("Decoded duration differs from the frozen timing manifest")
        if not all(np.isfinite(timing[key]) and timing[key] >= 0 for key in PHASES):
            raise ValueError("Nonfinite or negative timing")
        report["measured_raw"].append({**call, "corpus": clip["corpus"], "split": clip["split"],
                                        "label_for_sampling_only": clip["label"], "speaker_id": clip["speaker_id"],
                                        "prediction": prediction, "timing": timing, "audio": audio})
    report["measured_loop_wall_seconds"] = time.perf_counter() - started
    report["runtime_after"] = runtime(predictor)
    report["summary"] = grouped_summary(report["measured_raw"])
    medians = []
    for sample_id, clip in clips.items():
        calls = [row for row in report["measured_raw"] if row["sample_id"] == sample_id]
        if len(calls) != 3:
            raise ValueError("Expected three measurements per clip")
        medians.append({"sample_id": sample_id, "corpus": clip["corpus"],
                        "timing": {key: float(np.median([row["timing"][key] for row in calls])) for key in calls[0]["timing"]}})
    report["per_clip_medians"] = medians
    report["per_clip_median_summary"] = grouped_summary(medians)
    report["summary_interpretation"] = "Primary percentiles pool 120 timed calls (three per each of forty clips). Secondary percentiles use the forty per-clip medians. No accuracy metric or speaker-population uncertainty is estimated here."
    write_json(args.output, report)
    # A compatible single-clip view uses measurements already collected above;
    # it does not add a differently warmed benchmark run.
    canonical_id = plan["canonical_presentation_sample_id"]
    canonical = [row for row in report["measured_raw"] if row["sample_id"] == canonical_id]
    single = {"scope": "Canonical validation clip extracted from the prespecified multiclip benchmark",
              "model_kind": predictor.kind, "artifacts": predictor.artifacts,
              "initialization_ms": predictor.initialization_ms, "audio": canonical[-1]["audio"],
              "prediction": canonical[-1]["prediction"], "warmup_count": 3, "measured_repetitions": 3,
              "warmup_raw": [row["timing"] for row in report["warmup_raw"]],
              "measured_raw": [row["timing"] for row in canonical],
              "measured_summary": summarize(canonical)["statistics"],
              "rtf_definition": report["rtf_definition"], "timing_scope": report["timing_scope"],
              "runtime_before": report["runtime_before"], "runtime_after": report["runtime_after"],
              "source_benchmark_path": str(Path(args.output).resolve()), "source_benchmark_sha256": sha256(args.output),
              "sample_id": canonical_id}
    write_json(Path(args.output).with_name(f"single_clip_{backend_key}_benchmark.json"), single)
    return report


def combine(compact_path, emotion2vec_path, output, compact_neural_path=None):
    paths = [compact_path] + ([compact_neural_path] if compact_neural_path else []) + [emotion2vec_path]
    models = [json.loads(Path(path).read_text()) for path in paths]
    if len({model["plan_sha256"] for model in models}) != 1:
        raise ValueError("Models were benchmarked with different clip selections or orders")
    if len({model["backend_key"] for model in models}) != len(models):
        raise ValueError("Backend result keys must be unique")
    if any(len(model["measured_raw"]) != 120 for model in models):
        raise ValueError("Incomplete model benchmark")
    report = {"schema": "ser_cpu_benchmark_comparison_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "plan_path": models[0]["plan_path"], "plan_sha256": models[0]["plan_sha256"],
              "timing_scope": models[0]["timing_scope"], "rtf_definition": models[0]["rtf_definition"],
              "design": "40 clips, 20/corpus, five/class/corpus, three warmups/model, three repetitions/clip, 120 measured calls/model, identical recorded shuffled order; backend processes execute sequentially",
              "comparability": "Both backends perform full-file processing on the same recordings. Different Python/backend package environments are reported. No CPU hardware isolation or microphone latency claim is made.",
              "models": {model["backend_key"]: model for model in models}}
    write_json(output, report)
    lines = ["| Model | Corpus | Clips | Calls | Ready-file p50 (ms) | p95 (ms) | RTF p50 | RTF p95 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        for corpus in ["pooled", "RAVDESS", "TESS"]:
            summary = model["summary"][corpus]
            timing, rtf = summary["statistics"]["total_ready_file_ms"], summary["statistics"]["processing_rtf"]
            lines.append(f"| {model['model_label']} | {corpus} | {summary['unique_clips']} | {summary['measured_calls']} | {timing['p50']:.2f} | {timing['p95']:.2f} | {rtf['p50']:.4f} | {rtf['p95']:.4f} |")
    lines.extend(["", "Full-utterance processing from an already available audio file; excludes recording, startup, model initialization and UI. Percentiles summarize repeated calls, not microphone latency or generalization uncertainty. Project training/extraction workload status and hardware/thread details are recorded in the JSON. No hardware isolation was enforced.", ""])
    Path(output).with_suffix(".md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--seed", type=int, default=20260914)
    for name in ["run-compact", "run-emotion2vec"]:
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--model-label", required=True)
        command.add_argument("--workload-note", required=True)
        if name == "run-compact":
            command.add_argument("--backend-key", choices=["compact", "compact_neural"], default="compact")
            command.add_argument("--model", required=True)
            command.add_argument("--feature-metadata", required=True)
            command.add_argument("--normalizer")
            command.add_argument("--frozen-selections")
        else:
            command.add_argument("--encoder", required=True)
            command.add_argument("--head", required=True)
    combined = commands.add_parser("combine")
    combined.add_argument("--compact", required=True)
    combined.add_argument("--compact-neural")
    combined.add_argument("--emotion2vec", required=True)
    combined.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.seed)
    elif args.command == "combine":
        combine(args.compact, args.emotion2vec, args.output, args.compact_neural)
    else:
        run_backend(args)
    print(str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
