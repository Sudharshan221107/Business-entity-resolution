"""
Unit Test Suite for Business Entity Resolution.
Tests preprocessing, blocking, feature extraction, evaluation metrics, and validation.
"""
import unittest
import os
import sys

# Add src to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Fallback for scikit-learn in local environment if needed
local_venv = r"C:\Users\jegan\Downloads\AquaCycleML (1)\AquaCycleML\.venv\Lib\site-packages"
if os.path.exists(local_venv) and local_venv not in sys.path:
    sys.path.append(local_venv)

from preprocessing import (
    preprocess_business_name,
    preprocess_business_address,
    preprocess_country,
    preprocess_record,
    extract_numbers
)
from blocking import BlockingEngine
from features import (
    compute_levenshtein_distance,
    normalized_levenshtein_sim,
    jaccard_similarity,
    FeatureExtractor
)
from model import compute_entity_f05, compute_macro_f05

class TestPreprocessing(unittest.TestCase):
    def test_name_normalization(self):
        raw = "Apex Global Logistics, Inc."
        clean = preprocess_business_name(raw)
        self.assertIn("apex", clean)
        self.assertIn("global", clean)
        self.assertIn("inc", clean)

    def test_french_accents(self):
        raw = "Société Générale de Transport Routier SARL"
        clean = preprocess_business_name(raw)
        self.assertIn("societe", clean)
        self.assertIn("generale", clean)
        self.assertIn("sarl", clean)

    def test_address_normalization(self):
        raw = "1200 Industrial Pkwy, Suite 300, Near SBI ATM"
        clean = preprocess_business_address(raw)
        self.assertNotIn("near", clean.split())
        self.assertIn("1200", clean)
        self.assertIn("300", clean)

    def test_open_set_country(self):
        self.assertEqual(preprocess_country("US"), "us")
        self.assertEqual(preprocess_country("India"), "india")
        self.assertEqual(preprocess_country("France"), "france")
        self.assertEqual(preprocess_country(" Germany "), "germany")

    def test_number_extraction(self):
        nums = extract_numbers("Plot 45, Sector 18, Pin 201301")
        self.assertEqual(nums, ["45", "18", "201301"])

class TestSimilarityFeatures(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(compute_levenshtein_distance("apple", "apple"), 0)
        self.assertEqual(compute_levenshtein_distance("kitten", "sitting"), 3)
        self.assertAlmostEqual(normalized_levenshtein_sim("tech corp", "tech corp"), 1.0)
        self.assertLess(normalized_levenshtein_sim("apple", "orange"), 0.5)

    def test_jaccard(self):
        s1 = {"apex", "logistics"}
        s2 = {"apex", "logistics"}
        self.assertAlmostEqual(jaccard_similarity(s1, s2), 1.0)
        s3 = {"blue", "horizon"}
        self.assertAlmostEqual(jaccard_similarity(s1, s3), 0.0)

class TestF05Metric(unittest.TestCase):
    def test_singleton_correct(self):
        # True is empty, pred is empty -> 1.0
        score = compute_entity_f05(set(), set())
        self.assertEqual(score, 1.0)

    def test_singleton_false_alarm(self):
        # True is empty, pred is non-empty -> 0.0
        score = compute_entity_f05(set(), {"S2-00001"})
        self.assertEqual(score, 0.0)

    def test_missed_match(self):
        # True has matches, pred is empty -> 0.0
        score = compute_entity_f05({"S2-00001"}, set())
        self.assertEqual(score, 0.0)

    def test_perfect_match(self):
        # True matches pred -> 1.0
        score = compute_entity_f05({"S2-00001", "S3-00001"}, {"S2-00001", "S3-00001"})
        self.assertAlmostEqual(score, 1.0)

    def test_precision_heavy(self):
        # Example from challenge PDF:
        # Predicts [S2-00047, S2-00193, S3-00812]
        # Ground truth [S2-00047, S3-00812]
        # P = 2/3 (0.667), R = 1.0 -> F0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        score = compute_entity_f05(true, pred)
        self.assertAlmostEqual(score, 0.7142857, places=4)

class TestBlocking(unittest.TestCase):
    def test_blocking_recall(self):
        engine = BlockingEngine()
        target = [
            preprocess_record({"entity_id": "S2-00001", "business_name": "Apex Global Logistics", "business_address": "1200 Industrial Pkwy", "country": "US"}),
            preprocess_record({"entity_id": "S3-00001", "business_name": "Apex Global Logistics LLC", "business_address": "1200 Industrial Parkway", "country": "US"}),
            preprocess_record({"entity_id": "S2-00002", "business_name": "Boulangerie Saint-Germain", "business_address": "14 Rue de Rennes", "country": "France"}),
        ]
        engine.fit(target)

        query = preprocess_record({"entity_id": "S1-00001", "business_name": "Apex Glbl Logistics Inc", "business_address": "1200 Industrial Pkwy Ste 300", "country": "US"})
        candidates = engine.generate_candidates_for_record(query)

        self.assertIn("S2-00001", candidates)
        self.assertIn("S3-00001", candidates)
        # Should not include France record
        self.assertNotIn("S2-00002", candidates)

if __name__ == "__main__":
    unittest.main()
