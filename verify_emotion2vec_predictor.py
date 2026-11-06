"""One training-clip cache/adapter parity check and invalid-audio input checks.

The requested sample must belong to the training partition. No test rows are
predicted, no training occurs, and synthetic guard fixtures are not SER results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from emotion2vec_predictor_v10 import CLASSES, Emotion2vecPredictor, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("model", "head", "manifest", "features", "sample-id", "out"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    with open(args.manifest, newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["sample_id"] == args.sample_id]
    if len(matches) != 1 or matches[0]["split"] != "train":
        raise ValueError("This check is restricted to one uniquely identified training sample")
    row = matches[0]
    if sha256(row["path"]) != row["sha256"]:
        raise ValueError("Training audio differs from manifest")
    with np.load(args.features, allow_pickle=False) as archive:
        matches = np.flatnonzero(archive["sample_ids"] == args.sample_id)
        if len(matches) != 1:
            raise ValueError("Expected one matching cached embedding")
        cached = np.array(archive["embeddings"][matches[0]], copy=True)
    predictor = Emotion2vecPredictor(args.model, args.head)
    original_generate = predictor.encoder.generate
    captured = []

    def capture(*positional, **kwargs):
        result = original_generate(*positional, **kwargs)
        captured.append(np.array(result[0]["feats"], dtype=np.float32, copy=True))
        return result

    predictor.encoder.generate = capture
    try:
        prediction, timing, audio_info = predictor.predict_file(row["path"])
    finally:
        predictor.encoder.generate = original_generate
    if len(captured) != 1:
        raise AssertionError("Expected exactly one encoder call for one training file")
    embedding_match = bool(np.allclose(cached, captured[0], rtol=1e-5, atol=1e-6))
    if predictor.head_kind == "logistic":
        expected_scores = predictor.head.predict_proba(cached[None, :])[0]
        expected_label = int(predictor.head.classes_[expected_scores.argmax()])
    else:
        expected_scores = predictor.head.decision_function(cached[None, :])[0]
        expected_label = int(predictor.head.predict(cached[None, :])[0])
    actual_scores = np.array([prediction["scores"][label] for label in CLASSES])
    score_match = bool(np.allclose(expected_scores, actual_scores, rtol=1e-5, atol=1e-6))
    label_match = prediction["label_id"] == expected_label
    guards = {}
    # Each invalid input must fail before reaching the frozen encoder.
    def forbidden_generate(*args, **kwargs):
        raise AssertionError("Invalid audio reached the encoder")
    predictor.encoder.generate = forbidden_generate
    try:
        with tempfile.TemporaryDirectory(prefix="ser_invalid_audio_") as directory:
            fixtures = {"empty": np.empty(0, dtype=np.float32),
                        "zero": np.zeros(1000, dtype=np.float32),
                        "near_zero": np.full(1000, 1e-10, dtype=np.float32),
                        "nonfinite": np.full(1000, np.nan, dtype=np.float32)}
            for name, values in fixtures.items():
                path = Path(directory) / (name + ".wav")
                sf.write(path, values, 16000, subtype="FLOAT")
                try:
                    predictor.predict_file(path)
                except ValueError as error:
                    guards[name] = {"rejected_before_encoding": True, "message": str(error)}
                else:
                    guards[name] = {"rejected_before_encoding": False}
    finally:
        predictor.encoder.generate = original_generate
    passed = embedding_match and score_match and label_match and all(
        result["rejected_before_encoding"] for result in guards.values())
    report = {"status": "PASS" if passed else "FAIL",
              "scope": "One training-clip inference parity check; synthetic input guard fixtures; no test inference or model fitting",
              "sample_id": row["sample_id"], "split": row["split"],
              "embedding_matches_cache": embedding_match,
              "embedding_max_abs_error": float(np.max(np.abs(cached - captured[0]))),
              "head_scores_match_cache": score_match,
              "head_score_max_abs_error": float(np.max(np.abs(expected_scores - actual_scores))),
              "label_matches_cache": bool(label_match), "guard_checks": guards,
              "timing": timing, "audio_info": audio_info,
              "timing_caveat": "One cold training-file check under uncontrolled concurrent workload; not a latency benchmark.",
              "predictor_artifacts": predictor.artifacts,
              "verification_script_sha256": sha256(__file__)}
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "head_kind": predictor.head_kind,
                      "embedding_max_abs_error": report["embedding_max_abs_error"],
                      "head_score_max_abs_error": report["head_score_max_abs_error"], "out": args.out}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
