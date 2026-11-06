"""Frozen original emotion2vec base features and validation-only logistic head.

The encoder/checkpoint is external pretrained work by Ma et al. (ACL Findings 2024).
Model license and provenance must accompany the downloaded checkpoint.
This module never evaluates test labels or saves test predictions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

CLASSES = ["neutral", "happy", "sad", "angry"]
REVISION = "0c9a3152734f9d7a7a05b4ee6bfb3c109d288664"
CHECKPOINT_SHA = "4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd"
MODEL_ASSETS = {
    "config.yaml": "844e4728d4ca9ab355a2f77b905c7a2e74fe21ed3294444e88d1e52fd874f4ef",
    "configuration.json": "bf0717f50c1e4a016902f7a24b68f3023a1fc0fb539658a6072d61d6681fc57f",
}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def rows_from(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len({r["sample_id"] for r in rows})
    for r in rows:
        assert CLASSES[int(r["y"])] == r["label"]
    for corpus in {r["corpus"] for r in rows}:
        groups = [{r["speaker_id"] for r in rows if r["corpus"] == corpus and r["split"] == s}
                  for s in ("train", "val", "test")]
        assert not any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3))
    return rows

def audio(path):
    import librosa
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    assert np.isfinite(x).all() and len(x) > 400
    return np.ascontiguousarray(x, dtype="float32")

def load_encoder(directory, threads):
    import torch
    from funasr import AutoModel
    path = Path(directory)
    assert sha(path / "emotion2vec_base.pt") == CHECKPOINT_SHA
    for name, expected in MODEL_ASSETS.items():
        assert sha(path / name) == expected, f"Model configuration changed: {name}"
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    encoder = AutoModel(model=str(path.resolve()), device="cpu", ngpu=0,
                        disable_update=True, disable_pbar=True)
    encoder.model.eval()
    for p in encoder.model.parameters():
        p.requires_grad_(False)
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())
    return encoder

def embed(encoder, x):
    import numpy as np
    import torch
    with torch.inference_mode():
        result = encoder.generate(input=x, fs=16000, granularity="utterance",
                                  extract_embedding=True, disable_pbar=True)
    vector = np.asarray(result[0]["feats"], dtype="float32")
    assert vector.shape == (768,) and np.isfinite(vector).all()
    return vector

def benchmark(args):
    import numpy as np
    t = time.perf_counter()
    encoder = load_encoder(args.model, args.threads)
    load_s = time.perf_counter() - t
    x = audio(args.wav)
    times = []
    vectors = []
    for _ in range(4):
        t = time.perf_counter()
        vectors.append(embed(encoder, x))
        times.append(time.perf_counter() - t)
    assert np.allclose(vectors[-1], vectors[-2], rtol=1e-6, atol=1e-6)
    result = {"status": "measured_single_clip", "wav": str(Path(args.wav).resolve()),
              "audio_seconds": len(x)/16000, "model_load_seconds": load_s,
              "inference_seconds": times, "warm_median_seconds": float(np.median(times[1:])),
              "embedding_dimension": 768, "cpu_threads": args.threads,
              "checkpoint_sha256": CHECKPOINT_SHA,
              "estimated_2272_clips_seconds_from_this_clip": 2272*float(np.median(times[1:])),
              "estimate_caveat": "One clip; real corpus duration mix and concurrent load differ."}
    save_json(args.out, result)
    print(json.dumps(result), flush=True)

def extract(args):
    import importlib.metadata
    import numpy as np
    rows = rows_from(args.manifest)
    out = Path(args.out)
    cache = out.with_suffix("").with_name(out.stem + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    provenance = {"manifest_sha256": sha(args.manifest), "extractor_sha256": sha(__file__),
                  "model_revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA,
                  "encoder": "emotion2vec/emotion2vec_base", "embedding": "last-layer temporal mean, 768 dimensions",
                  "sample_rate": 16000, "resampler": "librosa soxr_hq", "channels": "mean to mono",
                  "waveform_normalization": "per-utterance torch layer_norm as specified by released normalize:true config",
                  "model_config_sha256": MODEL_ASSETS,
                  "upstream_fitted_here": False, "upstream_parameters_updated": False,
                  "test_labels_used": False, "classification_predictions_generated": False,
                  "cpu_threads": args.threads, "python": platform.python_version(),
                  "packages": {k: importlib.metadata.version(k) for k in ("torch", "torchaudio", "funasr", "numpy", "librosa", "soundfile")}}
    previous = cache / "provenance.json"
    if previous.exists():
        assert json.loads(previous.read_text()) == provenance, "Cache provenance changed"
    else:
        save_json(previous, provenance)
    encoder = load_encoder(args.model, args.threads)
    start = time.perf_counter()
    vectors, elapsed, durations = [], [], []
    for i, row in enumerate(rows):
        assert sha(row["path"]) == row["sha256"], f"Audio changed since manifest: {row['sample_id']}"
        p = cache / (hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".npz")
        if p.exists():
            d = np.load(p, allow_pickle=False)
            assert str(d["sample_id"]) == row["sample_id"]
            vector, seconds, duration = d["embedding"], float(d["seconds"]), float(d["duration"])
        else:
            t = time.perf_counter()
            x = audio(row["path"])
            vector = embed(encoder, x)
            seconds, duration = time.perf_counter()-t, len(x)/16000
            np.savez_compressed(p, sample_id=row["sample_id"], embedding=vector,
                                seconds=seconds, duration=duration)
        assert vector.shape == (768,) and np.isfinite(vector).all()
        vectors.append(vector)
        elapsed.append(seconds)
        durations.append(duration)
        if (i+1) % 100 == 0 or i == 0:
            print(json.dumps({"extracted": i+1, "total": len(rows), "elapsed_seconds": time.perf_counter()-start}), flush=True)
    np.savez_compressed(out, embeddings=np.stack(vectors),
                        sample_ids=np.array([r["sample_id"] for r in rows]),
                        processing_seconds=np.array(elapsed), audio_seconds=np.array(durations))
    provenance.update({"npz_sha256": sha(out), "n": len(rows), "all_features_finite": True,
                       "processing_seconds_sum": sum(elapsed), "audio_seconds_sum": sum(durations),
                       "processing_rtf": sum(elapsed)/sum(durations),
                       "p50_clip_processing_seconds": float(np.percentile(elapsed, 50)),
                       "p95_clip_processing_seconds": float(np.percentile(elapsed, 95)),
                       "timing_scope": "per-clip decode, mono/resample and frozen encoding; cold first clip included; model loading excluded"})
    save_json(out.with_suffix(".json"), provenance)
    print(json.dumps({"complete": str(out), "n": len(rows), "rtf": provenance["processing_rtf"]}), flush=True)

def fit(args):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rows = rows_from(args.manifest)
    feature_meta = json.loads(Path(args.features).with_suffix(".json").read_text())
    assert feature_meta["npz_sha256"] == sha(args.features), "Feature archive checksum mismatch"
    assert feature_meta["manifest_sha256"] == sha(args.manifest), "Feature manifest differs"
    assert feature_meta["model_revision"] == REVISION and feature_meta["checkpoint_sha256"] == CHECKPOINT_SHA
    assert feature_meta["encoder"] == "emotion2vec/emotion2vec_base"
    assert feature_meta["upstream_parameters_updated"] is False
    data = np.load(args.features, allow_pickle=False)
    assert data["sample_ids"].tolist() == [r["sample_id"] for r in rows]
    x = data["embeddings"]
    assert x.shape == (len(rows), 768) and np.isfinite(x).all()
    train = np.array([i for i,r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i,r in enumerate(rows) if r["split"] == "val"])
    ytrain = np.array([int(rows[i]["y"]) for i in train])
    yval = np.array([int(rows[i]["y"]) for i in val])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "selection.json").exists(), "Selection exists; preserve prior experiment"
    corpora = sorted({rows[i]["corpus"] for i in train})
    weights = np.empty(len(train), dtype=float)
    for corpus in corpora:
        for k in range(4):
            mask = np.array([rows[i]["corpus"] == corpus for i in train]) & (ytrain == k)
            assert mask.sum() > 0
            weights[mask] = len(train) / (len(corpora)*4*mask.sum())
    config = {"features_sha256": sha(args.features), "manifest_sha256": sha(args.manifest),
              "script_sha256": sha(__file__), "C_grid": [0.1,1.0,10.0], "labels": CLASSES,
              "feature_provenance_sha256": sha(Path(args.features).with_suffix(".json")),
              "selection": "highest validation UAR; smaller C on exact tie", "scaler_fit": "train only, unweighted",
              "sample_weights": "N_train/(number_of_corpora*4*training_count_in_corpus_class)",
              "solver": "lbfgs", "max_iter": 2000, "tol": 1e-5, "random_state": 17,
              "test_predictions_computed": False, "n_train": len(train), "n_val": len(val),
              "external_pretraining": "LibriSpeech initialization + emotion2vec listed 262h upstream corpora; RAVDESS/TESS absent from upstream list",
              "upstream_revision": REVISION, "upstream_sha256": CHECKPOINT_SHA}
    save_json(out / "config.json", config)
    trials=[]
    for c in config["C_grid"]:
        model = Pipeline([("scaler", StandardScaler()),
                          ("classifier", LogisticRegression(C=c,solver="lbfgs",max_iter=2000,tol=1e-5,random_state=17))])
        t = time.perf_counter()
        model.fit(x[train], ytrain, classifier__sample_weight=weights)
        prob = model.predict_proba(x[val]); pred=model.classes_[prob.argmax(1)]
        assert model.classes_.tolist() == [0,1,2,3]
        trial = {"C": c, "validation_uar": float(recall_score(yval,pred,average="macro",labels=range(4),zero_division=0)),
                 "validation_accuracy": float(accuracy_score(yval,pred)),
                 "validation_macro_f1": float(f1_score(yval,pred,average="macro",labels=range(4),zero_division=0)),
                 "validation_confusion_matrix": confusion_matrix(yval,pred,labels=range(4)).tolist(),
                 "fit_and_validation_seconds": time.perf_counter()-t,
                 "converged": bool(np.max(model.named_steps["classifier"].n_iter_) < 2000)}
        assert trial["converged"], "Logistic solver did not converge"
        stem=f"C{c:g}"
        joblib.dump(model,out / f"{stem}.joblib")
        with open(out / f"{stem}_validation_predictions.csv", "w", newline="") as f:
            fields=["sample_id","corpus","speaker_id","y_true","y_pred"]+[f"p_{k}" for k in CLASSES]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for j,i in enumerate(val):
                record={k:rows[i][k] for k in ["sample_id","corpus","speaker_id"]}
                record.update(y_true=int(yval[j]),y_pred=int(pred[j]))
                record.update({f"p_{k}":float(prob[j,z]) for z,k in enumerate(CLASSES)})
                writer.writerow(record)
        trial["model_sha256"]=sha(out / f"{stem}.joblib")
        trials.append(trial)
        save_json(out / "validation_trials.json",trials)
        print(json.dumps(trial),flush=True)
    chosen=sorted(trials,key=lambda r:(-r["validation_uar"],r["C"]))[0]
    chosen.update({"selected_model":f"C{chosen['C']:g}.joblib", "test_predictions_computed":False,
                   "config_sha256":sha(out / "config.json"),"classes":CLASSES})
    save_json(out / "selection.json",chosen)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("benchmark","extract"):
        s=sub.add_parser(name);s.add_argument("--model",required=True);s.add_argument("--out",required=True)
        s.add_argument("--threads",type=int,default=2)
        if name=="benchmark":s.add_argument("--wav",required=True)
        else:s.add_argument("--manifest",required=True)
    s=sub.add_parser("fit");s.add_argument("--manifest",required=True);s.add_argument("--features",required=True);s.add_argument("--out",required=True)
    args=p.parse_args()
    globals()[args.command](args)

if __name__=="__main__":main()
