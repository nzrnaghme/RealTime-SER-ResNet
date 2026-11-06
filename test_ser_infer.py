"""Synthetic-only inference integration checks; not SER research results."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ser_infer as infer


class InferenceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ser_infer_synthetic_")
        cls.root = Path(cls.temporary.name)
        cls.audio_path = cls.root / "synthetic_stereo_24k.wav"
        time_axis = np.arange(24000, dtype=np.float32) / 24000
        audio = np.column_stack([0.1 * np.sin(2 * np.pi * 211 * time_axis),
                                 0.05 * np.sin(2 * np.pi * 233 * time_axis)])
        sf.write(cls.audio_path, audio, 24000, subtype="PCM_16")
        cls.metadata = cls.root / "feature_metadata.json"
        cls.metadata.write_text(json.dumps({"feature_configuration_sha256": infer.features.FEATURE_CONFIG_HASH}))
        cls.logmel, cls.acoustic = infer.features.features_for_file(cls.audio_path)
        rng = np.random.default_rng(173)
        x = rng.normal(size=(40, 252)).astype(np.float32)
        y = np.tile(np.arange(4), 10)
        cls.svm = make_pipeline(StandardScaler(), SVC(C=1.0, probability=False)).fit(x, y)
        cls.svm_path = cls.root / "synthetic.joblib"
        joblib.dump(cls.svm, cls.svm_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_svm_preserves_saved_pipeline_predictions_and_decision_values(self):
        predictor = infer.Predictor(self.svm_path, self.metadata)
        prediction, timing, info = predictor.predict_file(self.audio_path)
        expected_label = int(self.svm.predict(self.acoustic[None, :])[0])
        expected_scores = self.svm.decision_function(self.acoustic[None, :])[0]
        self.assertEqual(prediction["label_id"], expected_label)
        np.testing.assert_allclose(list(prediction["scores"].values()), expected_scores, rtol=1e-7, atol=1e-7)
        self.assertEqual(prediction["score_kind"], "svm_ovr_decision_values_not_probabilities")
        self.assertEqual(info["native_sample_rate_hz"], 24000)
        self.assertEqual(info["native_channels"], 2)
        self.assertEqual(info["duration_s"], 1.0)
        phases = ["load_decode_ms", "resample_feature_ms", "normalization_ms", "model_call_ms", "postprocess_ms"]
        self.assertAlmostEqual(sum(timing[key] for key in phases), timing["total_ready_file_ms"], places=8)
        self.assertAlmostEqual(timing["processing_rtf"], timing["total_ready_file_ms"] / 1000, places=10)

    def test_neural_matches_saved_normalization_and_eager_logits(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(173)
        inputs = tf.keras.Input((128, 128, 1))
        pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        model = tf.keras.Model(inputs, tf.keras.layers.Dense(4)(pooled))
        model_path = self.root / "synthetic.keras"
        model.save(model_path)
        mean = np.linspace(-40, 5, 128, dtype=np.float32).reshape(1, 128, 1, 1)
        std = np.linspace(1, 3, 128, dtype=np.float32).reshape(1, 128, 1, 1)
        normalizer = self.root / "normalizer.npz"
        np.savez(normalizer, mean=mean, std=std)
        predictor = infer.Predictor(model_path, self.metadata, normalizer, compiled=True)
        normalized = predictor.normalize(self.logmel, self.acoustic)
        expected = ((self.logmel[None, ...] - mean) / std).astype(np.float32)
        np.testing.assert_array_equal(normalized, expected)
        compiled_logits = predictor.model_call(normalized)
        eager_logits = np.asarray(model(expected, training=False))
        np.testing.assert_allclose(compiled_logits, eager_logits, rtol=1e-6, atol=1e-6)
        prediction, _, _ = predictor.predict_file(self.audio_path)
        np.testing.assert_allclose(list(prediction["scores"].values()), infer.softmax(eager_logits)[0], rtol=1e-6)
        self.assertEqual(prediction["score_kind"], "uncalibrated_softmax")

    def test_benchmark_retains_warmups_raw_measurements_and_truthful_scope(self):
        args = infer.parser().parse_args([str(self.audio_path), "--model", str(self.svm_path),
                                        "--feature-metadata", str(self.metadata), "--benchmark",
                                        "--warmups", "1", "--repeats", "2",
                                        "--workload-note", "Synthetic integration check during shared CPU work."])
        report = infer.run(args)
        self.assertEqual(len(report["warmup_raw"]), 1)
        self.assertEqual(len(report["measured_raw"]), 2)
        self.assertAlmostEqual(report["measured_summary"]["total_ready_file_ms"]["mean"],
                               np.mean([row["total_ready_file_ms"] for row in report["measured_raw"]]))
        self.assertIn("not first-prediction or microphone latency", report["rtf_definition"])
        self.assertIn("No CPU isolation", report["runtime_after"]["cpu_isolation"])
        json.dumps(report, allow_nan=False)

    def test_feature_configuration_mismatch_is_rejected(self):
        invalid = self.root / "wrong_features.json"
        invalid.write_text(json.dumps({"feature_configuration_sha256": "mismatch"}))
        with self.assertRaisesRegex(ValueError, "feature metadata"):
            infer.Predictor(self.svm_path, invalid)

    def test_silent_recording_has_no_emotion_prediction(self):
        silent = self.root / 'silent.wav'
        sf.write(silent, np.zeros(16000, dtype=np.float32), 16000)
        predictor = infer.Predictor(self.svm_path, self.metadata)
        with self.assertRaisesRegex(ValueError, 'no usable audio signal'):
            predictor.predict_file(silent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
