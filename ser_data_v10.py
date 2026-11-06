"""Auditable four-class RAVDESS/TESS inventory and offline acoustic features.

This is a NEW experiment, not a reconstruction of the historical paper results.
Full-utterance temporal interpolation is deliberately NOT a 2-second streaming
feature pipeline. No training-fitted normalization or model training occurs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import librosa
import numpy as np
import soundfile as sf

CLASSES = ["neutral", "happy", "sad", "angry"]
RAV_EMOTIONS = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}
RAV_PATTERN = re.compile(
    r"^(03)-(01)-(0[1-8])-(0[12])-(0[12])-(0[12])-(\d{2})\.wav$", re.I
)
TESS_PATTERN = re.compile(
    r"^(OAF|YAF)_([A-Za-z][A-Za-z0-9_-]*)_"
    r"(neutral|happy|sad|angry|disgust|fear|fearful|ps|pleasant_surprise)\.wav$",
    re.I,
)
FEATURE_CONFIG = {
    "schema": "ser_features_v10.1",
    "temporal_scope": "full original utterance; offline classification",
    "sample_rate": 16000,
    "mono": "arithmetic mean across native channels before resampling",
    "resample": "librosa.resample res_type=soxr_hq, fix=True, scale=False",
    "amplitude_normalization": "none",
    "pre_emphasis": "none",
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "window": "hamming; scipy periodic convention through librosa",
    "center": True,
    "pad_mode": "constant",
    "n_mels": 128,
    "fmin": 0.0,
    "fmax": 8000.0,
    "mel_htk": False,
    "mel_norm": "slaney",
    "power": 2.0,
    "log_power": "10*log10, ref=1.0, amin=1e-10, top_db=80",
    "time_interpolation": "linear interpolation over full spectrogram to 128 uniformly spaced bins; endpoint-preserving",
    "logmel_shape": [128, 128, 1],
    "mfcc": "40 coefficients including c0 from native-length log-Mel; dct_type=2, norm=ortho, lifter=0",
    "delta": "width=9; orders1,2; mode=nearest; before time interpolation",
    "pooling": "mean/std per MFCC coefficient and each delta order (population std)",
    "pitch": "librosa.yin fmin=65,fmax=500,frame_length=1024,hop_length=160,center=True,pad_mode=constant",
    "pitch_gate": "RMS>=0.1*utterance_max_RMS and RMS>1e-8 and spectral_flatness<0.5; an acoustic heuristic, not ground-truth voicing",
    "pitch_pooling": "mean,std,p10,p50,p90 over gated frames; zero if no accepted frames",
    "energy": "20*log10(max(RMS,1e-8)); frame_length=1024,hop_length=160,center=True,pad_mode=constant; mean,std,p10,p50,p90",
    "additional_features": ["pitch_gate_fraction", "duration_s"],
    "acoustic_dimension": 252,
    "dtype": "float32",
    "learned_normalization": "NONE; fit standardization on training rows downstream",
}


class InventoryError(ValueError):
    """The files cannot safely form the declared manifest."""


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


FEATURE_CONFIG_HASH = _json_hash(FEATURE_CONFIG)


def _dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError("Explicit columns are required for an empty CSV")
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_ravdess(path: str | Path) -> dict | None:
    """Return selected canonical audio-only speech metadata, else None."""
    path = Path(path)
    match = RAV_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    actor = int(match[7])
    label = RAV_EMOTIONS.get(match[3])
    if not 1 <= actor <= 24 or label is None:
        return None
    # RAVDESS has no strong-neutral recordings; do not accept an impossible ID.
    if label == "neutral" and match[4] != "01":
        return None
    sample_id = "RAVDESS:" + path.stem.lower()
    return {
        "sample_id": sample_id,
        "corpus": "RAVDESS",
        "speaker_id": f"RAVDESS:{actor:02d}",
        "parent_utterance_id": sample_id,
        "label": label,
        "y": CLASSES.index(label),
        "path": str(path.resolve()),
        "split": "train" if actor <= 16 else "val" if actor <= 20 else "test",
        "fold": "A",
    }


def parse_tess(path: str | Path) -> dict | None:
    """TESS speaker is the original filename prefix, never the emotion folder."""
    path = Path(path)
    match = TESS_PATTERN.fullmatch(path.name)
    if match is None or match[3].lower() not in CLASSES:
        return None
    speaker, word, label = match[1].upper(), match[2].lower(), match[3].lower()
    sample_id = f"TESS:{speaker}_{word}_{label}"
    return {
        "sample_id": sample_id,
        "corpus": "TESS",
        "speaker_id": "TESS:" + speaker,
        "parent_utterance_id": sample_id,
        "label": label,
        "y": CLASSES.index(label),
        "path": str(path.resolve()),
        "split": "train" if speaker == "YAF" else "test",
        "fold": "A",
    }


def _audio_identity(path: Path) -> dict:
    """Hash native decoded samples, rate and shape, so container copies dedup."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all() or sr <= 0:
        raise InventoryError(f"Empty/nonfinite/invalid audio: {path}")
    pcm = np.ascontiguousarray(audio, dtype="<f4")
    # Canonicalize signed zero without mutating source files.
    pcm[pcm == 0] = 0.0
    header = json.dumps({"sample_rate": int(sr), "shape": list(pcm.shape)}, sort_keys=True).encode()
    waveform_hash = hashlib.sha256(header + b"\0" + pcm.tobytes()).hexdigest()
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "sha256": file_hash,
        "waveform_sha256": waveform_hash,
        "duration_s": len(audio) / float(sr),
        "native_sample_rate": int(sr),
        "native_channels": int(audio.shape[1]),
        "native_frames": int(len(audio)),
    }


def _root_list(values: Iterable[str | Path] | str | Path | None) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    return sorted({Path(value).resolve() for value in values})


def validate_rows(rows: list[dict], require_complete: bool = True) -> list[dict]:
    if not rows:
        raise InventoryError("No eligible four-class originals found")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise InventoryError("Manifest sample IDs are not unique")
    groups = {split: {r["speaker_id"] for r in rows if r["split"] == split} for split in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise InventoryError(f"Speaker overlap between {a} and {b}")
    # Parent grouping remains an independent invariant if derived rows are added later.
    for key in ("parent_utterance_id", "waveform_sha256"):
        seen = {}
        for row in rows:
            old_split = seen.setdefault(row[key], row["split"])
            if old_split != row["split"]:
                raise InventoryError(f"{key} crosses partitions: {row['sample_id']}")
    counts = []
    expected_speakers = [("RAVDESS", f"RAVDESS:{a:02d}") for a in range(1, 25)] + [("TESS", "TESS:OAF"), ("TESS", "TESS:YAF")]
    for corpus, speaker in expected_speakers:
        split = ("test" if speaker == "TESS:OAF" else "train") if corpus == "TESS" else ("train" if int(speaker[-2:]) <= 16 else "val" if int(speaker[-2:]) <= 20 else "test")
        for label in CLASSES:
            actual = sum(r["speaker_id"] == speaker and r["label"] == label for r in rows)
            expected = 200 if corpus == "TESS" else 4 if label == "neutral" else 8
            counts.append({"fold": "A", "split": split, "corpus": corpus, "speaker_id": speaker, "label": label, "n": actual, "expected_canonical_n": expected})
            if require_complete and actual != expected:
                raise InventoryError(f"Incomplete/noncanonical inventory: {speaker} {label}: {actual}, expected {expected}")
    if require_complete:
        for split in groups:
            if {r["label"] for r in rows if r["split"] == split} != set(CLASSES):
                raise InventoryError(f"{split} lacks classes")
    return counts


def build_inventory(ravdess_roots, tess_roots, out: str | Path, require_complete: bool = True) -> list[dict]:
    """Write an immutable fold-A manifest; explicit identical-copy dedup only.

    Existing nonempty output folders are rejected to preserve frozen inventories.
    Identical decoded audio with a different label, speaker or split is rejected.
    Canonical counts are validated BEFORE same-speaker/same-label/same-split
    waveform aliases are collapsed to their lexicographically first identity.
    Both the complete canonical inventory and explicit alias log are preserved.
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    roots = {"RAVDESS": _root_list(ravdess_roots), "TESS": _root_list(tess_roots)}
    excluded, duplicates, conflicts, rows, aliases = [], [], [], [], []
    by_id, by_wave, visited = {}, {}, set()
    try:
        for corpus, paths in roots.items():
            parser = parse_ravdess if corpus == "RAVDESS" else parse_tess
            for root in paths:
                if not root.is_dir():
                    raise FileNotFoundError(f"Data root is not a directory: {root}")
                for path in sorted(root.rglob("*")):
                    if not path.is_file() or path.suffix.lower() != ".wav":
                        continue
                    resolved = path.resolve()
                    if (corpus, resolved) in visited:
                        duplicates.append({"path": str(resolved), "kept_path": str(resolved), "sample_id": "", "reason": "same physical path encountered through overlapping roots"})
                        continue
                    visited.add((corpus, resolved))
                    row = parser(path)
                    if row is None:
                        excluded.append({"path": str(resolved), "corpus_root": corpus, "reason": "filename/modality/label not eligible canonical four-class original"})
                        continue
                    row.update(_audio_identity(path))
                    prior = by_id.get(row["sample_id"])
                    if prior:
                        if prior["waveform_sha256"] != row["waveform_sha256"]:
                            conflicts.append({"sample_id": row["sample_id"], "path": row["path"], "other_path": prior["path"], "reason": "same identity, different decoded audio"})
                            raise InventoryError(f"Conflicting audio for one identity: {row['sample_id']}")
                        duplicates.append({"path": row["path"], "kept_path": prior["path"], "sample_id": row["sample_id"], "reason": "same canonical identity and identical decoded waveform"})
                        continue
                    prior_wave = by_wave.get(row["waveform_sha256"])
                    if prior_wave:
                        reason = None
                        if prior_wave["label"] != row["label"]:
                            reason = "identical decoded audio has conflicting labels"
                        elif prior_wave["speaker_id"] != row["speaker_id"]:
                            reason = "identical decoded audio has conflicting speaker identities"
                        elif prior_wave["split"] != row["split"]:
                            reason = "identical decoded audio crosses partitions"
                        if reason:
                            conflicts.append({"sample_id": row["sample_id"], "path": row["path"], "other_path": prior_wave["path"], "reason": reason})
                            raise InventoryError(f"{reason}: {row['sample_id']} / {prior_wave['sample_id']}")
                    by_id[row["sample_id"]] = row
                    by_wave[row["waveform_sha256"]] = row
                    rows.append(row)
        rows.sort(key=lambda row: row["sample_id"])
        _write_csv(out / "excluded.csv", excluded, ["path", "corpus_root", "reason"])
        _write_csv(out / "duplicates.csv", duplicates, ["path", "kept_path", "sample_id", "reason"])
        _write_csv(out / "conflicts.csv", conflicts, ["sample_id", "path", "other_path", "reason"])
        if rows:
            _write_csv(out / "manifest_candidate.csv", rows)
        canonical_rows = list(rows)
        canonical_counts = validate_rows(canonical_rows, require_complete=require_complete)
        _write_csv(out / "canonical_manifest.csv", canonical_rows)
        _write_csv(out / "canonical_split_counts.csv", canonical_counts)
        unique_rows, representatives = [], {}
        for row in canonical_rows:  # Already sorted by sample ID: deterministic representative.
            kept = representatives.get(row["waveform_sha256"])
            if kept is None:
                representatives[row["waveform_sha256"]] = row
                unique_rows.append(row)
            else:
                aliases.append({"removed_sample_id": row["sample_id"], "kept_sample_id": kept["sample_id"],
                                "speaker_id": row["speaker_id"], "label": row["label"], "split": row["split"],
                                "removed_path": row["path"], "kept_path": kept["path"],
                                "removed_sha256": row["sha256"], "kept_sha256": kept["sha256"],
                                "waveform_sha256": row["waveform_sha256"],
                                "reason": "same speaker/label/split and identical waveform; lexicographically first canonical sample retained"})
        rows = unique_rows
        counts = validate_rows(rows, require_complete=False)
        _write_csv(out / "waveform_aliases.csv", aliases, ["removed_sample_id", "kept_sample_id", "speaker_id", "label", "split", "removed_path", "kept_path", "removed_sha256", "kept_sha256", "waveform_sha256", "reason"])
        _write_csv(out / "manifest.csv", rows)
        _write_csv(out / "split_counts.csv", counts)
        aggregate = []
        for split in ("train", "val", "test"):
            for corpus in ("RAVDESS", "TESS"):
                for label in CLASSES:
                    aggregate.append({"fold": "A", "split": split, "corpus": corpus, "label": label, "n": sum(r["split"] == split and r["corpus"] == corpus and r["label"] == label for r in rows)})
        _write_csv(out / "partition_counts.csv", aggregate)
        canonical = [{key: row[key] for key in ("sample_id", "parent_utterance_id", "corpus", "speaker_id", "label", "split", "fold", "waveform_sha256")} for row in rows]
        metadata = {
            "schema": "ser_inventory_v10.2", "status": "new_experiment_not_historical_reproduction",
            "classes": CLASSES, "n": len(rows), "require_complete": require_complete,
            "n_duplicates_deduplicated": len(duplicates), "n_excluded": len(excluded),
            "n_canonical_before_waveform_dedup": len(canonical_rows), "n_waveform_aliases_removed": len(aliases),
            "canonical_count_gate": "applied to canonical_manifest.csv before same-speaker/same-label/same-split waveform alias deduplication",
            "waveform_alias_rule": "retain lexicographically first sample ID; reject different speaker, label or partition; preserve waveform_aliases.csv",
            "roots": {key: [str(path) for path in value] for key, value in roots.items()},
            "inventory_content_sha256": _json_hash(canonical),
            "manifest_file_sha256": hashlib.sha256((out / "manifest.csv").read_bytes()).hexdigest(),
            "feature_configuration": FEATURE_CONFIG, "feature_configuration_sha256": FEATURE_CONFIG_HASH,
            "fold": {"id": "A", "RAVDESS_train": list(range(1, 17)), "RAVDESS_val": list(range(17, 21)), "RAVDESS_test": list(range(21, 25)), "TESS_train": ["YAF"], "TESS_val": [], "TESS_test": ["OAF"]},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "librosa": librosa.__version__, "soundfile": sf.__version__},
            "limitations": ["TESS has no validation speaker; all validation selection uses held-out RAVDESS speakers", "Full-utterance feature extraction is an offline benchmark, not the live-window system", "Canonical complete four-class inventory must be observed before claiming 2272 usable originals"],
        }
        _dump_json(out / "metadata.json", metadata)
        (out / "manifest_candidate.csv").unlink(missing_ok=True)
        return rows
    except Exception as exc:
        _write_csv(out / "excluded.csv", excluded, ["path", "corpus_root", "reason"])
        _write_csv(out / "duplicates.csv", duplicates, ["path", "kept_path", "sample_id", "reason"])
        _write_csv(out / "conflicts.csv", conflicts, ["sample_id", "path", "other_path", "reason"])
        _dump_json(out / "INVENTORY_FAILED.json", {"error": type(exc).__name__, "message": str(exc), "candidate_n": len(rows)})
        raise


def load_audio(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample_audio(audio, sr)


def resample_audio(audio_native: np.ndarray, sr: int) -> np.ndarray:
    """Downmix decoded (samples,channels) audio and convert to 16 kHz."""
    audio = np.asarray(audio_native, dtype=np.float32)
    if audio.ndim != 2 or not len(audio) or not audio.shape[1] or not np.isfinite(audio).all() or sr <= 0:
        raise ValueError("Expected nonempty finite (samples,channels) audio and a positive sample rate")
    mono = audio.mean(axis=1, dtype=np.float32)
    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000, res_type="soxr_hq", fix=True, scale=False)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _summary5(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return np.zeros(5, dtype=np.float32)
    return np.array([values.mean(), values.std(), *np.percentile(values, [10, 50, 90])], dtype=np.float32)


def acoustic_feature_names() -> list[str]:
    names = []
    for feature in ("mfcc", "delta1", "delta2"):
        for stat in ("mean", "std"):
            names.extend(f"{feature}_{coefficient:02d}_{stat}" for coefficient in range(40))
    names.extend(f"pitch_hz_{stat}" for stat in ("mean", "std", "p10", "p50", "p90"))
    names.extend(f"rms_db_{stat}" for stat in ("mean", "std", "p10", "p50", "p90"))
    names.extend(["pitch_gate_fraction", "duration_s"])
    return names


def features_for_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (128,128,1) log-Mel and (252,) acoustic statistics."""
    return features_for_audio(load_audio(path))


def features_for_audio(audio_16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same extractor for a decoded/downmixed/resampled 16-kHz mono waveform."""
    audio = np.asarray(audio_16k, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected a nonempty finite 16-kHz mono waveform")
    power = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=512, win_length=400, hop_length=160,
        window="hamming", center=True, pad_mode="constant", power=2.0,
        n_mels=128, fmin=0.0, fmax=8000.0, htk=False, norm="slaney", dtype=np.float32,
    )
    logmel_native = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=80.0)
    source_axis = np.linspace(0.0, 1.0, logmel_native.shape[1])
    target_axis = np.linspace(0.0, 1.0, 128)
    logmel = np.stack([np.interp(target_axis, source_axis, band) for band in logmel_native]).astype(np.float32)[..., None]
    mfcc = librosa.feature.mfcc(S=logmel_native, n_mfcc=40, dct_type=2, norm="ortho", lifter=0)
    parts = [mfcc, librosa.feature.delta(mfcc, width=9, order=1, mode="nearest"), librosa.feature.delta(mfcc, width=9, order=2, mode="nearest")]
    vector = [stat for part in parts for stat in (part.mean(axis=1), part.std(axis=1))]
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=160, center=True, pad_mode="constant")[0]
    energy_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    spectrum = np.abs(librosa.stft(audio, n_fft=1024, win_length=1024, hop_length=160, window="hann", center=True, pad_mode="constant"))
    flatness = librosa.feature.spectral_flatness(S=spectrum, power=2.0)[0]
    if float(np.max(np.abs(audio))) > 1e-8:
        pitch = librosa.yin(audio, fmin=65.0, fmax=500.0, sr=16000, frame_length=1024, hop_length=160, center=True, pad_mode="constant")
        if not (len(pitch) == len(rms) == len(flatness)):
            raise ValueError("Acoustic frame alignment mismatch")
        gate = (rms >= 0.1 * rms.max()) & (rms > 1e-8) & (flatness < 0.5) & np.isfinite(pitch)
        pitch_stats = _summary5(pitch[gate])
        gate_fraction = float(gate.mean())
    else:
        pitch_stats = np.zeros(5, dtype=np.float32)
        gate_fraction = 0.0
    vector.extend([pitch_stats, _summary5(energy_db), np.array([gate_fraction, len(audio) / 16000.0], dtype=np.float32)])
    acoustic = np.concatenate(vector).astype(np.float32)
    if logmel.shape != (128, 128, 1) or acoustic.shape != (252,) or not (np.isfinite(logmel).all() and np.isfinite(acoustic).all()):
        raise ValueError(f"Invalid feature output: {logmel.shape}, {acoustic.shape}")
    return logmel, acoustic


def extract_features(rows: list[dict], out: str | Path, workers: int = 2) -> dict:
    """Cache all features in exact manifest row order without fitting a scaler."""
    if not rows:
        raise ValueError("Cannot extract an empty manifest")
    if not 1 <= workers <= 4:
        raise ValueError("Use 1–4 CPU extraction workers")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "features.npz").exists():
        raise FileExistsError(f"Feature cache already exists: {out / 'features.npz'}")
    logmel = np.empty((len(rows), 128, 128, 1), dtype=np.float32)
    acoustic = np.empty((len(rows), 252), dtype=np.float32)
    # Warm lazy imports/Numba in one thread before launching modest concurrency.
    logmel[0], acoustic[0] = features_for_file(rows[0]["path"])
    print(f"FEATURES 1/{len(rows)}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(features_for_file, [row["path"] for row in rows[1:]])
            for index, (mel_row, acoustic_row) in enumerate(results, start=1):
                logmel[index], acoustic[index] = mel_row, acoustic_row
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(f"FEATURES {index + 1}/{len(rows)}", flush=True)
        temporary = out / "features_incomplete.npz"
        np.savez_compressed(temporary, logmel=logmel, acoustic=acoustic,
                            y=np.array([int(row["y"]) for row in rows], dtype=np.int64),
                            sample_ids=np.array([row["sample_id"] for row in rows], dtype=str),
                            feature_config_sha256=np.array(FEATURE_CONFIG_HASH),
                            class_names=np.array(CLASSES, dtype=str))
        temporary.replace(out / "features.npz")
        info = {"n": len(rows), "logmel_shape": list(logmel.shape), "acoustic_shape": list(acoustic.shape),
                "feature_configuration": FEATURE_CONFIG, "feature_configuration_sha256": FEATURE_CONFIG_HASH,
                "acoustic_feature_names": acoustic_feature_names(), "workers": workers,
                "features_sha256": hashlib.sha256((out / "features.npz").read_bytes()).hexdigest(),
                "row_order": "exactly manifest.csv order; verify sample_ids before training",
                "normalization": "none fitted here; downstream scaler must fit training rows only"}
        _dump_json(out / "feature_metadata.json", info)
        return info
    except Exception as exc:
        _dump_json(out / "FEATURES_FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravdess", nargs="+", required=True, help="One or more roots containing canonical RAVDESS WAVs")
    parser.add_argument("--tess", nargs="+", required=True, help="One or more roots containing canonical TESS WAVs")
    parser.add_argument("--out", required=True, help="New/empty output directory")
    parser.add_argument("--allow-incomplete", action="store_true", help="For explicitly labeled partial/synthetic checks; disables canonical count gate")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rows = build_inventory(args.ravdess, args.tess, args.out, require_complete=not args.allow_incomplete)
    print(json.dumps({"inventory_n": len(rows), "classes": CLASSES, "fold": "A", "complete_inventory_required": not args.allow_incomplete}), flush=True)
    if not args.inventory_only:
        info = extract_features(rows, args.out, workers=args.workers)
        print(json.dumps({"features": str(Path(args.out) / "features.npz"), "logmel_shape": info["logmel_shape"], "acoustic_shape": info["acoustic_shape"]}), flush=True)


if __name__ == "__main__":
    main()
