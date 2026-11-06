#!/usr/bin/env python3
"""Download and verify the exact original emotion2vec base used in SER v10.

Standard library only. Existing mismatched files are never overwritten.
Official URLs and hashes are copied from the verified acquisition record.
The model license URL is mutable; changed license bytes intentionally fail
verification rather than silently changing the recorded licensing provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.error
import urllib.request

MODEL_ID = "emotion2vec/emotion2vec_base"
REVISION = "0c9a3152734f9d7a7a05b4ee6bfb3c109d288664"
SOURCE = "https://huggingface.co/emotion2vec/emotion2vec_base"
METADATA_SOURCE = "https://huggingface.co/api/models/emotion2vec/emotion2vec_base?blobs=true"
LICENSE_SOURCE = "https://raw.githubusercontent.com/modelscope/FunASR/main/MODEL_LICENSE"
ASSETS = [
    {"file": "README.md", "bytes": 4011,
     "sha256": "fb643fa2195dcaefcc4cfacdaadcc2860300e1b2022f710548bfed9a559ebcd3"},
    {"file": "config.yaml", "bytes": 3028,
     "sha256": "844e4728d4ca9ab355a2f77b905c7a2e74fe21ed3294444e88d1e52fd874f4ef"},
    {"file": "configuration.json", "bytes": 302,
     "sha256": "bf0717f50c1e4a016902f7a24b68f3023a1fc0fb539658a6072d61d6681fc57f"},
    {"file": "emotion2vec_base.pt", "bytes": 1125606009,
     "sha256": "4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd"},
    {"file": "MODEL_LICENSE", "bytes": 5306,
     "sha256": "7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8"},
]

def asset_url(name: str) -> str:
    if name == "MODEL_LICENSE":
        return LICENSE_SOURCE
    return f"{SOURCE}/resolve/{REVISION}/{name}"

def verify_file(path: Path, record: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Existing or downloaded file size mismatch: {path}; no overwrite performed")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}; no overwrite performed")

def verify_acquisition(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular acquisition record: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_id") != MODEL_ID or data.get("revision") != REVISION:
        raise ValueError("Existing acquisition.json identifies a different model or revision")
    records = data.get("files", [])
    if len(records) != len({r.get("file") for r in records}):
        raise ValueError("Duplicate filenames in acquisition.json")
    by_name = {r.get("file"): r for r in records}
    for expected in ASSETS:
        got = by_name.get(expected["file"], {})
        if any(got.get(k) != expected[k] for k in ("bytes", "sha256")):
            raise ValueError(f"Existing acquisition.json has a different hash/size: {expected['file']}")
    return data

def install_without_overwrite(temporary: Path, target: Path, record: dict) -> None:
    """Hard-link a verified sibling temporary file atomically, without replacement."""
    try:
        os.link(temporary, target)
    except FileExistsError:
        verify_file(target, record)
    except OSError as exc:
        raise OSError(f"Cannot safely install {target} without overwrite; use a filesystem supporting hard links") from exc

def download_asset(directory: Path, record: dict, timeout: float, attempts: int) -> None:
    target = directory / record["file"]
    if target.exists() or target.is_symlink():
        verify_file(target, record)
        return
    for attempt in range(1, attempts + 1):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=record["file"] + ".", suffix=".part",
                                             dir=directory, delete=False) as destination:
                temporary = Path(destination.name)
                request = urllib.request.Request(asset_url(record["file"]),
                                                 headers={"User-Agent": "SER-v10-reproducible-downloader/1.0"})
                written = 0
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > record["bytes"]:
                            raise ValueError(f"Server returned more bytes than pinned asset {record['file']}")
                        destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            verify_file(temporary, record)
            install_without_overwrite(temporary, target, record)
            print(f"Verified {record['file']} ({record['bytes']:,} bytes)", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"Download failed after {attempts} attempts: {record['file']}") from exc
            print(f"Retrying {record['file']} after network failure ({attempt}/{attempts})", file=sys.stderr)
            time.sleep(min(2 ** attempt, 8))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

def write_acquisition(directory: Path) -> None:
    target = directory / "acquisition.json"
    if target.exists() or target.is_symlink():
        verify_acquisition(target)
        print("Preserved existing verified acquisition.json", flush=True)
        return
    data = {
        "model_id": MODEL_ID, "revision": REVISION, "source": SOURCE,
        "metadata_source": METADATA_SOURCE,
        "weights_verified_against_hub_lfs_sha256": True,
        "files": ASSETS,
        "license_source": LICENSE_SOURCE,
        "license_attribution": "Emotion2vec by Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang, Xie Chen; FunASR model license from Alibaba Group.",
        "upstream_provenance_status": "Published named corpora exclude RAVDESS and TESS; independent full upstream recording-level audit not available.",
    }
    contents = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    record = {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="acquisition.", suffix=".part", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        install_without_overwrite(temporary, target, record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("emotion2vec_base"))
    parser.add_argument("--verify-only", action="store_true", help="No network access or writes; require all assets and acquisition.json")
    parser.add_argument("--timeout", type=float, default=60.0, help="Socket timeout in seconds")
    parser.add_argument("--attempts", type=int, default=3, help="Bounded retries for network errors")
    args = parser.parse_args()
    if args.timeout <= 0 or not 1 <= args.attempts <= 10:
        parser.error("timeout must be positive and attempts must be between 1 and 10")
    directory = args.output.resolve()
    if args.verify_only:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    else:
        directory.mkdir(parents=True, exist_ok=True)
    acquisition = directory / "acquisition.json"
    if acquisition.exists() or acquisition.is_symlink():
        verify_acquisition(acquisition)
    elif args.verify_only:
        raise FileNotFoundError(acquisition)
    # Check every existing destination before requesting any missing asset.
    for record in ASSETS:
        path = directory / record["file"]
        if path.exists() or path.is_symlink():
            verify_file(path, record)
        elif args.verify_only:
            raise FileNotFoundError(path)
    if not args.verify_only:
        for record in ASSETS:
            download_asset(directory, record, args.timeout, args.attempts)
        for record in ASSETS:
            verify_file(directory / record["file"], record)
        write_acquisition(directory)
    print(json.dumps({"status": "verified", "model": MODEL_ID, "revision": REVISION,
                      "directory": str(directory), "verify_only": args.verify_only}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
