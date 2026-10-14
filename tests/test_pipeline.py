"""Meaningful checks for raw features, label validation and saved-model reuse."""

import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from sms_classifier import (
    build_pipeline, extract_spam_features, load_dataset, setup_nltk,
)


class PipelineTests(unittest.TestCase):
    def test_raw_signals_survive(self):
        text = "FREE £500!!! Visit HTTPS://example.com"
        features = extract_spam_features(text)
        self.assertEqual(features["exclamation_count"], 3)
        self.assertEqual(features["digit_count"], 3)
        self.assertEqual(features["has_url"], 1)
        self.assertEqual(features["has_money_or_bait_word"], 1)
        self.assertEqual(extract_spam_features("123!!!")["uppercase_ratio"], 0)

    def test_duplicates_removed_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "spam.csv"
            pd.DataFrame({"label": ["ham", "ham", "spam"],
                          "message": ["hello", "hello", "free prize"]}).to_csv(path, index=False)
            self.assertEqual(len(load_dataset(path)), 2)
            pd.DataFrame({"label": ["ham", "spam"],
                          "message": ["same", "same"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_dataset(path)

    def test_saved_pipeline_accepts_raw_text(self):
        setup_nltk()
        frame = load_dataset(Path(__file__).resolve().parents[1] / "data" / "spam.csv")
        sample, _, target, _ = train_test_split(
            frame.message, frame.target, train_size=300, stratify=frame.target, random_state=42,
        )
        model = build_pipeline().fit(sample, target)
        messages = ["Hi, are we meeting tomorrow?", "Free prize! Call 12345 now!"]
        before = model.predict(messages)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            reloaded = joblib.load(path)
            self.assertListEqual(before.tolist(), reloaded.predict(messages).tolist())


if __name__ == "__main__":
    unittest.main()
