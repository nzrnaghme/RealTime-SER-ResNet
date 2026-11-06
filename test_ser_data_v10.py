"""Synthetic provenance/parser checks; never SER performance experiments."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import ser_data_v10 as data


def tone(path, frequency=220.0, length=3200, subtype="PCM_16"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = 0.2 * np.sin(2 * np.pi * frequency * np.arange(length) / 16000)
    sf.write(path, audio, 16000, subtype=subtype)


class DataTests(unittest.TestCase):
    def test_canonical_parsers_and_speaker_folds(self):
        for actor, split in [(1, "train"), (16, "train"), (17, "val"), (20, "val"), (21, "test"), (24, "test")]:
            row = data.parse_ravdess(f"03-01-03-02-01-02-{actor:02d}.wav")
            self.assertEqual((row["corpus"], row["speaker_id"], row["label"], row["split"]), ("RAVDESS", f"RAVDESS:{actor:02d}", "happy", split))
            self.assertEqual(row["sample_id"], row["parent_utterance_id"])
        for name in ["03-02-03-01-01-01-01.wav", "01-01-03-01-01-01-01.wav", "03-01-02-01-01-01-01.wav", "03-01-01-02-01-01-01.wav", "03-01-03-01-01-01-25.wav", "03-01-03-01-01-01-01_aug.wav"]:
            self.assertIsNone(data.parse_ravdess(name), name)
        self.assertEqual(data.parse_tess("OAF_back_angry.wav")["speaker_id"], "TESS:OAF")
        self.assertEqual(data.parse_tess("YAF_back_neutral.wav")["split"], "train")
        for name in ["OAF_back_ps.wav", "YAF_back_fear.wav", "OTHER_back_angry.wav", "YAF_back_angry_aug.wav"]:
            self.assertIsNone(data.parse_tess(name), name)

    def test_deduplicate_identical_identity_and_waveform_but_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "rav" / "a" / "03-01-03-01-01-01-01.wav"
            tone(original)
            duplicate = root / "rav" / "b" / original.name
            duplicate.parent.mkdir()
            # Same decoded samples, different WAV encoding/container bytes.
            decoded, rate = sf.read(original, dtype="float32")
            sf.write(duplicate, decoded, rate, subtype="FLOAT")
            self.assertNotEqual(original.read_bytes(), duplicate.read_bytes())
            (root / "tess").mkdir()
            tone(root / "tess" / "YAF_back_angry.wav", frequency=330)
            rows = data.build_inventory(root / "rav", root / "tess", root / "out", require_complete=False)
            self.assertEqual(len(rows), 2)
            with (root / "out" / "duplicates.csv").open() as handle:
                duplicate_rows = list(csv.DictReader(handle))
            self.assertEqual(len(duplicate_rows), 1)
            self.assertIn("identical decoded", duplicate_rows[0]["reason"])
            metadata = json.loads((root / "out" / "metadata.json").read_text())
            self.assertEqual(metadata["n_duplicates_deduplicated"], 1)
            self.assertEqual(len(metadata["inventory_content_sha256"]), 64)

    def test_reject_different_waveforms_with_one_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = "03-01-03-01-01-01-01.wav"
            tone(root / "rav" / "a" / filename, frequency=220)
            tone(root / "rav" / "b" / filename, frequency=330)
            (root / "tess").mkdir()
            with self.assertRaisesRegex(data.InventoryError, "Conflicting audio"):
                data.build_inventory(root / "rav", root / "tess", root / "out", require_complete=False)
            self.assertTrue((root / "out" / "conflicts.csv").exists())
            self.assertFalse((root / "out" / "manifest.csv").exists())

    def test_reject_identical_audio_with_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone(root / "tess" / "OAF_back_angry.wav")
            shutil.copy2(root / "tess" / "OAF_back_angry.wav", root / "tess" / "OAF_back_sad.wav")
            (root / "rav").mkdir()
            with self.assertRaisesRegex(data.InventoryError, "conflicting labels"):
                data.build_inventory(root / "rav", root / "tess", root / "out", require_complete=False)

    def test_canonical_repetition_alias_keeps_first_id_and_records_both(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "rav" / "03-01-03-01-02-01-07.wav"
            second = root / "rav" / "03-01-03-01-02-02-07.wav"
            tone(first)
            shutil.copy2(first, second)
            (root / "tess").mkdir()
            rows = data.build_inventory(root / "rav", root / "tess", root / "out", require_complete=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "RAVDESS:" + first.stem)
            with (root / "out" / "canonical_manifest.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with (root / "out" / "waveform_aliases.csv").open() as handle:
                aliases = list(csv.DictReader(handle))
            self.assertEqual(aliases[0]["removed_sample_id"], "RAVDESS:" + second.stem)
            self.assertEqual(aliases[0]["kept_sample_id"], "RAVDESS:" + first.stem)

    def test_same_waveform_with_another_speaker_is_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone(root / "tess" / "OAF_back_angry.wav")
            shutil.copy2(root / "tess" / "OAF_back_angry.wav", root / "tess" / "YAF_back_angry.wav")
            (root / "rav").mkdir()
            with self.assertRaisesRegex(data.InventoryError, "conflicting speaker"):
                data.build_inventory(root / "rav", root / "tess", root / "out", require_complete=False)

    def test_complete_inventory_gate_cannot_silently_pass_partial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone(root / "rav" / "03-01-03-01-01-01-01.wav")
            (root / "tess").mkdir()
            with self.assertRaisesRegex(data.InventoryError, "Incomplete/noncanonical"):
                data.build_inventory(root / "rav", root / "tess", root / "out")
            self.assertFalse((root / "out" / "manifest.csv").exists())

    def test_features_silence_and_entire_utterance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silence = root / "silence.wav"
            sf.write(silence, np.zeros(1600, dtype=np.float32), 16000)
            mel, acoustic = data.features_for_file(silence)
            self.assertEqual(mel.shape, (128, 128, 1))
            self.assertEqual(acoustic.shape, (252,))
            self.assertTrue(np.isfinite(mel).all() and np.isfinite(acoustic).all())
            np.testing.assert_array_equal(acoustic[240:245], np.zeros(5))
            self.assertEqual(acoustic[250], 0.0)
            # Same first half, different second half: output must see beyond a start crop.
            t = np.arange(16000) / 16000
            first = 0.2 * np.sin(2 * np.pi * 220 * t)
            second = 0.2 * np.sin(2 * np.pi * 700 * t)
            sf.write(root / "a.wav", np.concatenate([first, first]), 16000)
            sf.write(root / "b.wav", np.concatenate([first, second]), 16000)
            a, _ = data.features_for_file(root / "a.wav")
            b, _ = data.features_for_file(root / "b.wav")
            self.assertGreater(float(np.abs(a[:, 80:] - b[:, 80:]).mean()), 0.1)
            self.assertEqual(len(data.acoustic_feature_names()), 252)

    def test_feature_cache_preserves_manifest_row_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone(root / "OAF_back_angry.wav", frequency=220)
            tone(root / "YAF_back_happy.wav", frequency=330)
            # Deliberately non-sorted order to catch accidental feature reordering.
            rows = [data.parse_tess(root / "YAF_back_happy.wav"), data.parse_tess(root / "OAF_back_angry.wav")]
            info = data.extract_features(rows, root / "cache", workers=2)
            with np.load(root / "cache" / "features.npz", allow_pickle=False) as cache:
                self.assertEqual(cache["sample_ids"].tolist(), [row["sample_id"] for row in rows])
                self.assertEqual(cache["y"].tolist(), [1, 3])
                self.assertEqual(cache["logmel"].shape, (2, 128, 128, 1))
                self.assertEqual(cache["acoustic"].shape, (2, 252))
                self.assertEqual(cache["feature_config_sha256"].item(), data.FEATURE_CONFIG_HASH)
            self.assertEqual(len(info["acoustic_feature_names"]), 252)


if __name__ == "__main__":
    unittest.main(verbosity=2)
