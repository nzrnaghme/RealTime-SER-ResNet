#!/usr/bin/env python3
"""Auditable CPU SER experiments. Training never evaluates the test partition.

Commands: train -> freeze -> evaluate. Final evaluation requires an explicit frozen
selection file. Acoustic and log-Mel features cover the same full utterances.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ["neutral", "happy", "sad", "angry"]
CONFIGS = {
    "cnn": {"residual": False, "clip": False, "scheduler": False},
    "r0": {"residual": True, "clip": False, "scheduler": False},
    "r1": {"residual": True, "clip": True, "scheduler": False},
    "r2": {"residual": True, "clip": False, "scheduler": True},
    "r3": {"residual": True, "clip": True, "scheduler": True},
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data(features_path, manifest_path):
    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    with np.load(features_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("logmel", "acoustic", "y", "sample_ids")}
    n = len(rows)
    if any(len(value) != n for value in data.values()):
        raise ValueError("Feature arrays and manifest have different row counts")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != n or ids != data["sample_ids"].tolist():
        raise ValueError("Manifest IDs must be unique and match feature IDs in order")
    if data["logmel"].shape[1:] != (128, 128, 1):
        raise ValueError("Expected logmel shape N x 128 x 128 x 1")
    if data["acoustic"].ndim != 2 or not np.array_equal(data["y"], [int(row["y"]) for row in rows]):
        raise ValueError("Acoustic shape or label alignment invalid")
    if set(np.unique(data["y"])) != set(range(4)):
        raise ValueError("Expected all four label IDs 0,1,2,3")
    groups = {}
    for row in rows:
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        for field in ("speaker_id", "parent_utterance_id"):
            key = (field, row["corpus"], row[field])
            groups.setdefault(key, set()).add(row["split"])
    if any(len(value) != 1 for value in groups.values()):
        raise ValueError("Speaker or original utterance crosses partitions")
    indices = {split: np.array([i for i, row in enumerate(rows) if row["split"] == split])
               for split in ("train", "val", "test")}
    for split, idx in indices.items():
        if not len(idx) or set(np.unique(data["y"][idx])) != set(range(4)):
            raise ValueError(f"{split} must have all four classes")
    return data, rows, indices


def metrics(y, pred):
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "uar": float(np.mean(r)), "macro_f1": float(np.mean(f)),
        "weighted_f1": float(np.average(f, weights=support)),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(4)).tolist(),
        "classes": {name: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(support[i])}
                    for i, name in enumerate(LABELS)},
    }


def grouped_metrics(y, pred, rows):
    result = {"pooled": metrics(y, pred), "corpus": {}, "speaker": {}}
    for field, key in (("corpus", "corpus"), ("speaker_id", "speaker")):
        names = sorted({r[field] for r in rows})
        for name in names:
            mask = np.array([row[field] == name for row in rows])
            result[key][name] = metrics(y[mask], pred[mask])
    result["corpus_balanced_uar"] = float(np.mean([m["uar"] for m in result["corpus"].values()]))
    return result


def predictions(path, rows, y, pred, scores, score_kind):
    output = []
    for i, row in enumerate(rows):
        record = {key: row[key] for key in ("sample_id", "corpus", "speaker_id", "split")}
        record.update(true_label=LABELS[int(y[i])], predicted_label=LABELS[int(pred[i])], score_kind=score_kind)
        record.update({f"score_{label}": float(scores[i, j]) for j, label in enumerate(LABELS)})
        output.append(record)
    write_csv(path, output)


def weights_for_train(y, rows, balance):
    if balance == "none":
        return np.ones(len(y), dtype="float32")
    corpora = sorted({row["corpus"] for row in rows})
    weights = np.empty(len(y), dtype="float32")
    for corpus in corpora:
        for label in range(4):
            mask = np.array([row["corpus"] == corpus for row in rows]) & (y == label)
            count = int(mask.sum())
            if count == 0:
                raise ValueError("Corpus-class balancing requires every train corpus/class cell")
            weights[mask] = len(y) / (len(corpora) * 4 * count)
    return weights


def tensorflow():
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        pass
    return tf


def build_model(residual, seed=17):
    tf = tensorflow()
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    inputs = layers.Input((128, 128, 1), name="logmel")
    x = layers.Conv2D(16, 5, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPool2D(2, name="stem_pool")(x)
    for stage, width in enumerate((16, 32, 64)):
        prefix = f"stage{stage + 1}"
        if stage:
            # Shared transition in BOTH bodies; skip adds no parameters.
            x = layers.AveragePooling2D(2, name=f"{prefix}_pool")(x)
            x = layers.Conv2D(width, 1, use_bias=False, name=f"{prefix}_transition")(x)
            x = layers.BatchNormalization(name=f"{prefix}_transition_bn")(x)
            x = layers.Activation("relu", name=f"{prefix}_transition_relu")(x)
        identity = x
        x = layers.Conv2D(width, 3, padding="same", use_bias=False, name=f"{prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu1")(x)
        x = layers.Conv2D(width, 3, padding="same", use_bias=False, name=f"{prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
        if residual:
            x = layers.Add(name=f"{prefix}_identity_add")([identity, x])
        x = layers.Activation("relu", name=f"{prefix}_relu2")(x)
    x = layers.GlobalAveragePooling2D(name="global_average")(x)
    outputs = layers.Dense(4, name="emotion_logits")(x)
    return tf.keras.Model(inputs, outputs, name="residual_cnn" if residual else "plain_cnn")


def mel_normalize_fit(x):
    # Per-Mel-band statistics across training utterances and their time bins.
    mean = np.mean(x, axis=(0, 2), keepdims=True, dtype=np.float64).astype("float32")
    std = np.std(x, axis=(0, 2), keepdims=True, dtype=np.float64).astype("float32")
    return mean, np.maximum(std, 1e-6)


def infer_neural(model, x, batch_size=64):
    return np.concatenate([np.asarray(model(x[i:i + batch_size], training=False))
                           for i in range(0, len(x), batch_size)])


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(y, probabilities):
    return float(-np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean())


def train_neural(args, data, rows, idx, config, seed, out):
    tf = tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    train_ids, val_ids = idx["train"], idx["val"]
    mean, std = mel_normalize_fit(data["logmel"][train_ids])
    np.savez(out / "normalizer.npz", mean=mean, std=std)
    xtrain = ((data["logmel"][train_ids] - mean) / std).astype("float32")
    xval = ((data["logmel"][val_ids] - mean) / std).astype("float32")
    ytrain = data["y"][train_ids].astype("int64")
    yval = data["y"][val_ids].astype("int64")
    if not np.isfinite(xtrain).all() or not np.isfinite(xval).all():
        raise ValueError("Nonfinite train/validation features")
    weights = weights_for_train(ytrain, [rows[i] for i in train_ids], args.balance)
    model = build_model(config["residual"], seed)
    optargs = {"learning_rate": args.lr}
    if config["clip"]:
        optargs["global_clipnorm"] = 1.0
    optimizer = tf.keras.optimizers.Adam(**optargs)

    @tf.function(reduce_retracing=True)
    def train_step(x, y, weight):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(losses * weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        norm = tf.linalg.global_norm(gradients)
        tf.debugging.check_numerics(loss, "Nonfinite training loss")
        tf.debugging.check_numerics(norm, "Nonfinite preclip gradient norm")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        correct = tf.reduce_sum(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int64) == y, tf.float32))
        return loss, norm, correct

    best = (-1.0, -float("inf"))
    best_epoch = 0
    stop_wait = 0
    lr_wait = 0
    lr_best_uar = -1.0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(xtrain))
        losses, norms, batches, corrects = [], [], [], []
        lr_used = float(optimizer.learning_rate.numpy())
        for begin in range(0, len(order), args.batch_size):
            batch = order[begin:begin + args.batch_size]
            loss, norm, correct = train_step(xtrain[batch], ytrain[batch], weights[batch])
            losses.append(float(loss.numpy()))
            norms.append(float(norm.numpy()))
            corrects.append(float(correct.numpy()))
            batches.append(len(batch))
        probabilities = softmax(infer_neural(model, xval))
        val_loss = cross_entropy(yval, probabilities)
        val_metrics = metrics(yval, probabilities.argmax(axis=1))
        uar = val_metrics["uar"]
        criterion = (uar, -val_loss)
        if criterion > best:
            best, best_epoch, stop_wait = criterion, epoch, 0
            model.save(out / "best.keras")
        else:
            stop_wait += 1
        # Plateau uses strict validation UAR improvement; loss tie-break is checkpoint-only.
        if uar > lr_best_uar:
            lr_best_uar, lr_wait = uar, 0
        else:
            lr_wait += 1
        if config["scheduler"] and lr_wait >= args.scheduler_patience:
            optimizer.learning_rate.assign(max(args.min_lr, lr_used * 0.5))
            lr_wait = 0
        record = {
            "epoch": epoch, "train_weighted_loss": float(np.average(losses, weights=batches)),
            "train_accuracy": float(sum(corrects) / len(xtrain)),
            "val_loss": val_loss, "val_uar": uar, "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"], "lr_used": lr_used,
            "lr_next": float(optimizer.learning_rate.numpy()),
            "preclip_gradient_norm_mean": float(np.mean(norms)),
            "preclip_gradient_norm_p95": float(np.quantile(norms, .95)),
            "preclip_gradient_norm_max": float(np.max(norms)),
            "preclip_gradient_norm_fraction_above_1": float(np.mean(np.asarray(norms) > 1)),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(record)
        write_csv(out / "history.csv", history)
        print(f"{out.name} epoch={epoch:02d} val_UAR={uar:.4f} val_loss={val_loss:.4f} seconds={record['epoch_seconds']:.1f}", flush=True)
        if stop_wait >= args.patience:
            break
    elapsed = time.perf_counter() - started
    model = tf.keras.models.load_model(out / "best.keras", compile=False)
    probabilities = softmax(infer_neural(model, xval))
    val_rows = [rows[i] for i in val_ids]
    predictions(out / "validation_predictions.csv", val_rows, yval, probabilities.argmax(axis=1), probabilities, "softmax_uncalibrated")
    result = {
        "status": "complete", "model_kind": "neural", "seed": seed,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "parameter_count": int(model.count_params()), "training_seconds": elapsed,
        "validation": grouped_metrics(yval, probabilities.argmax(axis=1), val_rows),
        "selection_rule": "maximum validation UAR; lower validation CE breaks ties; earliest if equal",
        "checkpoint_sha256": sha256(out / "best.keras"),
        "normalizer_sha256": sha256(out / "normalizer.npz"),
    }
    write_json(out / "result.json", result)
    return result


def train_svm(args, data, rows, idx, out):
    train_ids, val_ids = idx["train"], idx["val"]
    xtrain, xval = data["acoustic"][train_ids], data["acoustic"][val_ids]
    ytrain, yval = data["y"][train_ids], data["y"][val_ids]
    if not np.isfinite(xtrain).all() or not np.isfinite(xval).all():
        raise ValueError("Nonfinite train/validation acoustic features")
    weights = weights_for_train(ytrain, [rows[i] for i in train_ids], args.balance)
    trials = []
    best = -1.0
    started = time.perf_counter()
    for c in (.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale", probability=False))
        model.fit(xtrain, ytrain, svc__sample_weight=weights)
        pred = model.predict(xval)
        trial = {"C": c, **metrics(yval, pred)}
        trials.append(trial)
        if trial["uar"] > best:  # Ascending C means smaller C wins exact ties.
            best, chosen_c = trial["uar"], c
            joblib.dump(model, out / "best.joblib")
    model = joblib.load(out / "best.joblib")
    pred = model.predict(xval)
    scores = model.decision_function(xval)
    val_rows = [rows[i] for i in val_ids]
    predictions(out / "validation_predictions.csv", val_rows, yval, pred, scores, "svm_ovr_decision_not_probability")
    result = {
        "status": "complete", "model_kind": "svm", "seed": None, "chosen_C": chosen_c,
        "C_grid": [.1, 1., 10.], "gamma": "scale", "training_seconds": time.perf_counter() - started,
        "validation": grouped_metrics(yval, pred, val_rows),
        "selection_rule": "maximum validation UAR; smaller C breaks ties",
        "checkpoint_sha256": sha256(out / "best.joblib"),
    }
    write_json(out / "svm_trials.json", trials)
    write_json(out / "result.json", result)
    print(f"svm chosen_C={chosen_c} val_UAR={best:.4f}", flush=True)
    return result


def run_train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FROZEN_SELECTIONS.json").exists():
        raise RuntimeError("Campaign is frozen. Use a new explicitly exploratory output directory for further development")
    data, rows, idx = load_data(args.features, args.manifest)
    setting = {
        "features_sha256": sha256(args.features), "manifest_sha256": sha256(args.manifest),
        "experiment_driver_sha256": sha256(__file__),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "tensorflow": tensorflow().__version__},
        "labels": LABELS, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "lr": args.lr, "balance": args.balance,
        "scheduler_patience": args.scheduler_patience, "scheduler_factor": .5, "min_lr": args.min_lr,
        "input_coverage": "full utterance; 128 temporal bins by deterministic resampling",
        "augmentation": "none", "normalization": "training-only, per Mel band / acoustic StandardScaler",
        "selection_data": "val only; no test predictions during train",
        "architecture": "stem Conv5/16 stride2, BN/ReLU, maxpool2; 16/32/64 stages, two Conv3 each; shared avgpool+Conv1 channel transitions; GAP, Dense4 logits",
        "residual_contrast": "identical learned layers/parameter counts; identity additions are the sole architectural difference",
        "early_stop_rule": "patience epochs without checkpoint improvement (UAR then CE tie-break)",
        "scheduler_rule": "factor 0.5 after patience epochs without strict validation-UAR improvement; zero cooldown",
        "class_corpus_weights": "N_train/(number_of_corpora*4*training_count_in_corpus_class)" if args.balance == "corpus-class" else "all 1",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "tensorflow_threads": 2},
    }
    existing = out / "campaign_config.json"
    if existing.exists() and json.loads(existing.read_text()) != setting:
        raise RuntimeError("Existing campaign settings differ; use a fresh output directory")
    write_json(existing, setting)
    planned = [name if name == "svm" else f"{name}_seed{seed}"
               for name in args.models for seed in ([None] if name == "svm" else args.seeds)]
    planpath = out / "run_plan.json"
    if planpath.exists():
        planned = sorted(set(planned) | set(json.loads(planpath.read_text())["planned_runs"]))
    write_json(planpath, {"planned_runs": planned})
    for name in args.models:
        if name not in CONFIGS and name != "svm":
            raise ValueError(name)
        for seed in ([None] if name == "svm" else args.seeds):
            run = out / (name if seed is None else f"{name}_seed{seed}")
            run.mkdir(exist_ok=True)
            if (run / "result.json").exists():
                print(f"Skipping completed run {run.name}", flush=True)
                continue
            if (run / "FAILED.json").exists():
                raise RuntimeError(f"Run {run.name} failed previously; preserve it and use a fresh campaign for a retry")
            config = {} if name == "svm" else CONFIGS[name]
            write_json(run / "config.json", {"name": name, "seed": seed, **config, **setting})
            try:
                if name == "svm":
                    train_svm(args, data, rows, idx, run)
                else:
                    train_neural(args, data, rows, idx, config, seed, run)
            except Exception as exc:
                write_json(run / "FAILED.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                print(f"FAILED {run.name}: {type(exc).__name__}: {exc}", flush=True)
                if args.fail_fast:
                    raise
    print("Training complete. No test metrics were computed. Review validation, then run freeze.", flush=True)


def run_freeze(args):
    root = Path(args.output)
    target = root / "FROZEN_SELECTIONS.json"
    if target.exists():
        raise RuntimeError("Selection already frozen")
    runs, failed = [], []
    for configpath in sorted(root.glob("*/config.json")):
        run = configpath.parent
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            runs.append({"run": run.name, "config_sha256": sha256(configpath),
                         "checkpoint_sha256": result["checkpoint_sha256"],
                         "normalizer_sha256": result.get("normalizer_sha256"),
                         "model_kind": result["model_kind"], "validation_uar": result["validation"]["pooled"]["uar"]})
        else:
            failed.append(run.name)
    if not runs:
        raise RuntimeError("No completed models to freeze")
    planned = json.loads((root / "run_plan.json").read_text())["planned_runs"]
    completed = {run["run"] for run in runs}
    failed = sorted(set(failed) | (set(planned) - completed))
    by_config = {}
    for run in runs:
        name = run["run"].split("_seed")[0]
        by_config.setdefault(name, []).append(run["validation_uar"])
    means = {key: float(np.mean(value)) for key, value in by_config.items()}
    # All runs are evaluated; this choice designates deployment only, never selects the best seed.
    frozen = {"campaign_config_sha256": sha256(root / "campaign_config.json"), "runs": runs,
              "failed_or_incomplete_runs": failed, "mean_validation_uar_by_config": means,
              "validation_selected_configuration": max(sorted(means), key=lambda name: means[name]),
              "test_not_used_for_selection": True,
              "evaluation_rule": "evaluate all frozen successful runs and retain failure list; no best-test-score selection",
              "frozen_unix_time": time.time()}
    write_json(target, frozen)
    print(f"Frozen {len(runs)} successful runs, {len(failed)} failed/incomplete. Configuration chosen by validation: {frozen['validation_selected_configuration']}")


def latency(fn, example, warmups=5, repetitions=30):
    for _ in range(warmups):
        fn(example)
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn(example)
        times.append((time.perf_counter() - start) * 1000)
    return {"scope": "ready-feature batch-one model call; excludes audio acquisition, feature extraction and UI",
            "warmups": warmups, "repetitions": repetitions, "p50_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, .95)), "raw_ms": times}


def run_evaluate(args):
    root = Path(args.output)
    frozen_path = Path(args.frozen or root / "FROZEN_SELECTIONS.json")
    frozen = json.loads(frozen_path.read_text())
    config = json.loads((root / "campaign_config.json").read_text())
    if sha256(root / "campaign_config.json") != frozen["campaign_config_sha256"]:
        raise RuntimeError("Campaign config changed after freeze")
    if sha256(args.features) != config["features_sha256"] or sha256(args.manifest) != config["manifest_sha256"]:
        raise RuntimeError("Evaluation data differs from training manifest/features")
    data, rows, idx = load_data(args.features, args.manifest)
    test_ids = idx["test"]
    y = data["y"][test_ids]
    test_rows = [rows[i] for i in test_ids]
    summary = []
    for entry in frozen["runs"]:
        run = root / entry["run"]
        if sha256(run / "config.json") != entry["config_sha256"]:
            raise RuntimeError(f"Run config changed: {run.name}")
        model_path = run / ("best.joblib" if entry["model_kind"] == "svm" else "best.keras")
        if sha256(model_path) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed: {run.name}")
        if entry["normalizer_sha256"] is not None and sha256(run / "normalizer.npz") != entry["normalizer_sha256"]:
            raise RuntimeError(f"Training normalizer changed: {run.name}")
        if entry["model_kind"] == "svm":
            model = joblib.load(model_path)
            x = data["acoustic"][test_ids]
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite test features")
            scores, pred = model.decision_function(x), model.predict(x)
            score_kind = "svm_ovr_decision_not_probability"
            timing = latency(model.predict, x[:1])
        else:
            tf = tensorflow()
            model = tf.keras.models.load_model(model_path, compile=False)
            with np.load(run / "normalizer.npz") as archive:
                mean, std = archive["mean"], archive["std"]
            x = ((data["logmel"][test_ids] - mean) / std).astype("float32")
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite test features")
            scores = softmax(infer_neural(model, x))
            pred = scores.argmax(axis=1)
            score_kind = "softmax_uncalibrated"
            timing = latency(lambda batch: np.asarray(model(batch, training=False)), x[:1])
        result = grouped_metrics(y, pred, test_rows)
        write_json(run / "test_metrics.json", result)
        write_json(run / "ready_feature_latency.json", timing)
        predictions(run / "test_predictions.csv", test_rows, y, pred, scores, score_kind)
        record = {"run": run.name, "configuration": run.name.split("_seed")[0],
                  "validation_uar": entry["validation_uar"], "test_accuracy": result["pooled"]["accuracy"],
                  "test_uar": result["pooled"]["uar"], "test_macro_f1": result["pooled"]["macro_f1"],
                  "test_corpus_balanced_uar": result["corpus_balanced_uar"]}
        for corpus, value in result["corpus"].items():
            for measure in ("accuracy", "uar", "macro_f1"):
                record[f"{corpus}_{measure}"] = value[measure]
        summary.append(record)
        print(f"FINAL {run.name}: accuracy={record['test_accuracy']:.4f}, UAR={record['test_uar']:.4f}, corpus-balanced UAR={record['test_corpus_balanced_uar']:.4f}", flush=True)
    write_csv(root / "test_summary.csv", summary)
    aggregates = {}
    for name in sorted({record["configuration"] for record in summary}):
        records = [record for record in summary if record["configuration"] == name]
        values = {}
        for key in records[0]:
            if isinstance(records[0][key], (float, int)):
                array = [record[key] for record in records]
                values[key] = {"mean": float(np.mean(array)), "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else None}
        aggregates[name] = {"successful_run_count": len(records), "scores": values}
    write_json(root / "test_aggregate.json", {"by_configuration": aggregates,
               "failed_or_incomplete_runs": frozen["failed_or_incomplete_runs"],
               "uncertainty_note": "Seed SD measures training variability, not speaker-population uncertainty. One TESS test speaker per fold.",
               "historical_comparison_note": "New protocol differs from historical 1685-item subset; no direct numerical improvement claim."})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    evaluate = sub.add_parser("evaluate")
    freeze = sub.add_parser("freeze")
    for command in (train, evaluate):
        command.add_argument("--features", required=True)
        command.add_argument("--manifest", required=True)
    for command in (train, evaluate, freeze):
        command.add_argument("--output", required=True)
    train.add_argument("--models", nargs="+", default=["svm", "cnn", "r0", "r1", "r2", "r3"])
    train.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--scheduler-patience", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=.001)
    train.add_argument("--min-lr", type=float, default=1e-7)
    train.add_argument("--balance", choices=["none", "corpus-class"], default="corpus-class")
    train.add_argument("--fail-fast", action="store_true")
    evaluate.add_argument("--frozen")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"train": run_train, "freeze": run_freeze, "evaluate": run_evaluate}[args.command](args)
