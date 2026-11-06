"""Read-only asset verification for the frozen emotion2vec experiment.

Reads source files, installed-package metadata and audio bytes. It does not load
a model, decode audio, extract features, train a classifier, or make predictions.
The only write is the requested JSON verification report.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

PINNED_REVISION = "0c9a3152734f9d7a7a05b4ee6bfb3c109d288664"
PINNED_CHECKPOINT = "4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--cache-provenance", required=True, type=Path)
    parser.add_argument("--extraction-log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    failures = []

    def require(condition: bool, description: str) -> None:
        if not condition:
            failures.append(description)

    acquisition_path = args.model / "acquisition.json"
    acquisition = json.loads(acquisition_path.read_text())
    metadata = json.loads(args.metadata.read_text())
    cache = json.loads(args.cache_provenance.read_text())
    require(acquisition["revision"] == metadata["sha"] == cache["model_revision"] == PINNED_REVISION,
            "Model revision does not match pinned upstream revision.")
    require(metadata["modelId"] == acquisition["model_id"] == "emotion2vec/emotion2vec_base",
            "Model ID differs from original emotion2vec base.")
    siblings = {item["rfilename"]: item for item in metadata["siblings"]}
    model_checks = []
    for item in acquisition["files"]:
        path = args.model / item["file"]
        require(path.resolve().is_relative_to(args.model.resolve()), "Acquisition file outside model directory.")
        check = {"file": item["file"], "bytes": path.stat().st_size,
                 "sha256": digest(path), "expected_sha256": item["sha256"]}
        check["matches_acquisition"] = check["sha256"] == item["sha256"] and check["bytes"] == item["bytes"]
        require(check["matches_acquisition"], f"Changed acquired asset: {item['file']}")
        upstream = siblings.get(item["file"])
        if upstream is not None:
            if "lfs" in upstream:
                check["matches_pinned_hub_object"] = check["sha256"] == upstream["lfs"]["sha256"]
            else:
                check["git_blob_sha1"] = git_blob_sha1(path)
                check["matches_pinned_hub_object"] = check["git_blob_sha1"] == upstream["blobId"]
            require(check["matches_pinned_hub_object"], f"Asset differs from pinned hub object: {item['file']}")
        else:
            check["upstream_object_note"] = "Separate license acquisition; integrity checked against acquisition SHA-256, not the checkpoint's hub revision."
        model_checks.append(check)
    check_by_name = {item["file"]: item for item in model_checks}
    for name in ("config.yaml", "configuration.json", "MODEL_LICENSE", "emotion2vec_base.pt"):
        require(name in check_by_name, f"Required acquired asset missing: {name}")
    require(check_by_name["emotion2vec_base.pt"]["sha256"] == cache["checkpoint_sha256"] == PINNED_CHECKPOINT,
            "Checkpoint SHA differs from pinned checkpoint or extraction cache.")

    configuration = json.loads((args.model / "configuration.json").read_text())
    file_metas = configuration["file_path_metas"]
    require(file_metas["config"] == "config.yaml" and file_metas["init_param"] == "emotion2vec_base.pt",
            "Model configuration does not resolve to verified config/checkpoint files.")
    import yaml
    config = yaml.safe_load((args.model / file_metas["config"]).read_text())
    require(config["model"] == "Emotion2vec" and config["model_conf"]["normalize"] is True,
            "Verified model config does not specify Emotion2vec with waveform normalization enabled.")

    extractor_source = args.extractor.read_text()
    module = ast.parse(extractor_source)
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("REVISION", "CHECKPOINT_SHA")}
    require(constants == {"REVISION": PINNED_REVISION, "CHECKPOINT_SHA": PINNED_CHECKPOINT},
            "Extractor constants differ from pinned upstream assets.")
    require(digest(args.extractor) == cache["extractor_sha256"], "Active extraction script differs from cached provenance.")
    loader = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "load_encoder")
    calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "AutoModel"]
    require(len(calls) == 1, "Expected one AutoModel construction in load_encoder.")
    loader_keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    require(loader_keywords.get("model") == "str(path.resolve())", "AutoModel does not use the supplied local model directory.")
    require(set(loader_keywords) == {"model", "device", "ngpu", "disable_update", "disable_pbar"},
            "Unexpected AutoModel kwargs may override verified configuration.")

    package_versions = {name: importlib.metadata.version(name) for name in cache["packages"]}
    require(package_versions == cache["packages"], "Installed dependency versions differ from extraction provenance.")
    distribution = importlib.metadata.distribution("funasr")
    installed_file = next(item for item in distribution.files if str(item) == "funasr/models/emotion2vec/model.py")
    source_path = distribution.locate_file(installed_file)
    source_sha = digest(source_path)
    source_text = source_path.read_text()
    recorded_hash = installed_file.hash
    record_match = recorded_hash is not None and recorded_hash.mode == "sha256" and (
        base64.urlsafe_b64encode(bytes.fromhex(source_sha)).decode().rstrip("=") == recorded_hash.value)
    require(record_match, "Installed emotion2vec implementation differs from its package RECORD.")
    normalize_statement = bool(re.search(r"if self\.cfg\.normalize:\s+source = F\.layer_norm\(source, source\.shape\)", source_text))
    require(normalize_statement, "Expected FunASR per-utterance waveform layer normalization not found.")
    require("feats = np.mean(feats, axis=0)" in source_text, "Expected utterance mean pooling not found.")

    manifest_sha = digest(args.manifest)
    require(manifest_sha == cache["manifest_sha256"], "Current manifest differs from active extraction provenance.")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len({row["sample_id"] for row in rows}), "Manifest IDs are not unique.")
    audio_checks = []
    for row in rows:
        actual = digest(Path(row["path"]))
        matches = actual == row["sha256"]
        require(matches, f"Changed audio: {row['sample_id']}")
        audio_checks.append({"sample_id": row["sample_id"], "sha256": actual, "matches_manifest": matches})
    group_counts = Counter((row["corpus"], row["split"]) for row in rows)
    log = args.extraction_log.read_text()
    load_path_evidence = f"Loading pretrained params from {(args.model / 'emotion2vec_base.pt').resolve()}" in log
    keys_evidence = "status: <All keys matched successfully>" in log
    require(load_path_evidence and keys_evidence, "Extraction log does not confirm verified checkpoint path and successful state-dictionary match.")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only on all existing assets; only this JSON report was written. No model loading, training, audio decoding, feature extraction or predictions.",
        "verification_script_sha256": digest(Path(__file__)),
        "failures": failures,
        "model_revision": PINNED_REVISION,
        "acquisition_sha256": digest(acquisition_path),
        "hub_metadata_sha256": digest(args.metadata),
        "model_assets": model_checks,
        "license_source": acquisition.get("license_source"),
        "license_scope_note": "Acquired license bytes match acquisition record; license came from a separate mutable main-branch URL and is not established as historical text at the checkpoint revision. No legal interpretation is performed.",
        "configuration_resolution": {"file_path_metas": file_metas, "loader_keywords": loader_keywords,
            "checkpoint_load_path_confirmed_in_log": load_path_evidence,
            "checkpoint_all_keys_matched_in_log": keys_evidence,
            "evidence_scope": "On-disk pinned asset hashes plus static extraction call/config resolution and existing loader log; no live process introspection."},
        "preprocessing_semantics": {"normalize": True,
            "waveform_normalization": "After mono conversion and 16-kHz resampling, FunASR applies torch.nn.functional.layer_norm over the complete individual waveform (default eps=1e-5). This is per utterance, with no fitted corpus statistics.",
            "embedding_pooling": "Temporal arithmetic mean of returned final-layer feature frames; 768 dimensions.",
            "utterance_scope": "Complete individual clip; not streaming or fixed first-output latency."},
        "installed_upstream_source": {"package": "funasr", "version": distribution.version,
            "file": str(source_path), "sha256": source_sha, "matches_package_record": record_match},
        "package_versions": package_versions,
        "manifest_sha256": manifest_sha,
        "audio_count": len(rows),
        "audio_count_by_corpus_split": [{"corpus": corpus, "split": split, "n": n}
                                         for (corpus, split), n in sorted(group_counts.items())],
        "all_current_audio_sha256_match_manifest": all(item["matches_manifest"] for item in audio_checks),
        "audio_checks": audio_checks,
        "cache_binding": {"provenance_sha256": digest(args.cache_provenance),
            "extractor_sha256": digest(args.extractor), "manifest_matches": manifest_sha == cache["manifest_sha256"],
            "note": "This companion report closes current-run asset checks; it does not change the extraction script's cache invalidation behavior."},
        "upstream_overlap_limit": acquisition.get("upstream_provenance_status"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "audio_verified": len(rows),
                      "model_assets_verified": len(model_checks), "failures": failures, "report": str(args.out)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
