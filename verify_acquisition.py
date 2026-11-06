"""Verify official public SER archives and extract only metadata-verified WAV files."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import zlib
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tess_metadata() -> dict:
    record = json.loads((ROOT / "tess_record.json").read_text())
    files = record["data"]["latestVersion"]["files"]
    result = {item["dataFile"]["filename"]: item["dataFile"] for item in files}
    assert len(result) == len(files) == 2800
    # The official download archive uses OAF_bite_neutral.wav, while the record
    # calls the same file OA_bite_neutral.wav. Its size and MD5 match exactly.
    # Preserve the downloaded spelling and retain the record name in provenance.
    if "OA_bite_neutral.wav" in result:
        assert "OAF_bite_neutral.wav" not in result
        result["OAF_bite_neutral.wav"] = result.pop("OA_bite_neutral.wav")
    return result


def audio_headers(paths: list[Path]) -> list[dict]:
    counts = Counter()
    for path in paths:
        info = sf.info(path)
        counts[(info.samplerate, info.channels, info.subtype, info.format)] += 1
    return [
        {"sample_rate_hz": rate, "channels": channels, "sample_subtype": subtype, "container": container, "files": count}
        for (rate, channels, subtype, container), count in sorted(counts.items())
    ]


def write_verified_tess(name: str, raw: bytes, metadata: dict) -> None:
    assert Path(name).name == name and name in metadata, name
    expected = metadata[name]
    assert len(raw) == expected["filesize"], name
    assert hashlib.md5(raw).hexdigest() == expected["checksum"]["value"], name
    destination = DATA / "TESS" / name
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(raw)


def recover_partial_tess(archive: Path) -> int:
    """Read complete ZIP entries from a truncated streaming ZIP; verify every WAV.

    This is only an optional transfer accelerator, not evidence that the archive
    is complete. A final metadata inventory verification remains mandatory.
    """
    metadata = tess_metadata()
    completed = 0
    with archive.open("rb") as stream:
        while True:
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                break
            values = struct.unpack("<4s5H3L2H", header)
            _, _, flags, method, _, _, crc, compressed_size, _, name_size, extra_size = values
            name = stream.read(name_size).decode("utf-8")
            stream.read(extra_size)
            assert method == 8 and flags & 8, (name, flags, method)
            inflater = zlib.decompressobj(-15)
            chunks = []
            while not inflater.eof:
                block = stream.read(65536)
                if not block:
                    return completed
                chunks.append(inflater.decompress(block))
            stream.seek(-len(inflater.unused_data), 1)
            descriptor = stream.read(16)
            assert len(descriptor) == 16 and descriptor[:4] == b"PK\x07\x08"
            _, expected_crc, expected_compressed_size, expected_size = struct.unpack("<4s3L", descriptor)
            raw = b"".join(chunks)
            assert len(raw) == expected_size and zlib.crc32(raw) == expected_crc, name
            write_verified_tess(name, raw, metadata)
            completed += 1
    return completed


def finish(tess_archive: Path) -> dict:
    rav_archive = DATA / "RAVDESS_official.zip"
    rav_md5 = digest(rav_archive, "md5")
    assert rav_md5 == "bc696df654c87fed845eb13823edef8a"
    rav_out = (DATA / "RAVDESS").resolve()
    rav_out.mkdir(exist_ok=True)
    with zipfile.ZipFile(rav_archive) as archive:
        for member in archive.infolist():
            assert (rav_out / member.filename).resolve().is_relative_to(rav_out)
        archive.extractall(rav_out)
    rav_files = sorted(rav_out.rglob("*.wav"))
    assert len(rav_files) == 1440
    assert len({path.name for path in rav_files}) == 1440
    assert all(path.name.startswith("03-01-") for path in rav_files)
    metadata = tess_metadata()
    seen = set()
    with zipfile.ZipFile(tess_archive) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if name == "MANIFEST.TXT":
                (ROOT / "TESS_download_manifest.txt").write_bytes(archive.read(member))
                continue
            assert name not in seen, name
            write_verified_tess(name, archive.read(member), metadata)
            seen.add(name)
    assert seen == set(metadata), sorted(set(metadata) - seen)
    tess_files = sorted((DATA / "TESS").glob("*.wav"))
    assert len(tess_files) == 2800
    checks = []
    for path in tess_files:
        item = metadata[path.name]
        actual_md5 = digest(path, "md5")
        assert actual_md5 == item["checksum"]["value"]
        checks.append({"filename": path.name, "metadata_filename": item["filename"],
                       "file_id": item["id"], "bytes": path.stat().st_size,
                       "md5": actual_md5, "official_md5": item["checksum"]["value"]})
    (ROOT / "tess_verified_files.json").write_text(json.dumps(checks, indent=2) + "\n")
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "official_public_repositories; not private Google Drive derivatives",
        "RAVDESS": {
            "record": "https://zenodo.org/records/1188976",
            "download": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
            "archive": str(rav_archive), "archive_bytes": rav_archive.stat().st_size,
            "official_archive_md5": "bc696df654c87fed845eb13823edef8a", "verified_archive_md5": rav_md5,
            "archive_sha256": digest(rav_archive, "sha256"),
            "path": str(rav_out), "wav_count": len(rav_files),
            "audio_header_counts": audio_headers(rav_files),
            "speakers": dict(sorted(Counter(path.stem.split("-")[-1] for path in rav_files).items())),
            "modality": "audio-only speech (03-01), all eight original emotion classes",
        },
        "TESS": {
            "record": "https://doi.org/10.5683/SP2/E8H2MF",
            "metadata": "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF",
            "download": "https://borealisdata.ca/api/access/dataset/93347",
            "dataset_version": "1.0", "archive": str(tess_archive),
            "license": json.loads((ROOT / "tess_record.json").read_text())["data"]["latestVersion"]["license"],
            "archive_bytes": tess_archive.stat().st_size, "archive_sha256": digest(tess_archive, "sha256"),
            "path": str(DATA / "TESS"), "wav_count": len(tess_files),
            "audio_header_counts": audio_headers(tess_files),
            "all_2800_file_md5_match_official_metadata": True,
            "filename_aliases": [{"archive_filename": "OAF_bite_neutral.wav",
                                  "metadata_filename": "OA_bite_neutral.wav", "file_id": 95758,
                                  "reason": "Official archive spelling differs from metadata; identical MD5 bc32cdf920bb82376687fe228c56bd30 verifies identity."}],
            "checksums_file": str(ROOT / "tess_verified_files.json"),
            "speakers": dict(sorted(Counter(path.name.split("_")[0] for path in tess_files).items())),
            "notes": "All seven original categories acquired; four target classes selected downstream.",
        },
    }
    (ROOT / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path)
    parser.add_argument("--tess-archive", type=Path, default=DATA / "TESS_official_complete.zip")
    args = parser.parse_args()
    if args.partial:
        print("Recovered and MD5 verified:", recover_partial_tess(args.partial))
    else:
        print(json.dumps(finish(args.tess_archive), indent=2))
