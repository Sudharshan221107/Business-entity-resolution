"""
End-to-End Business Entity Resolution Pipeline.
Orchestrates data loading, preprocessing, blocking, feature extraction,
training, threshold optimization, test inference, output generation, and validation.
"""
import os
import csv
import random
import numpy as np

from .preprocessing import preprocess_record
from .blocking import BlockingEngine
from .features import FeatureExtractor
from .model import EntityMatchingModel, compute_macro_f05

def read_tsv_dict(filepath: str) -> list[dict]:
    """Reads TSV file with explicit tab separator into a list of row dicts."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            records.append(row)
    return records

def read_ground_truth(filepath: str) -> dict[str, list[str]]:
    """Reads ground truth TSV: source1_entity_id -> list of matched IDs."""
    gt = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            matched_str = row[1].strip() if len(row) > 1 else ""
            matched_ids = [m.strip() for m in matched_str.split(',') if m.strip()]
            gt[s1_id] = matched_ids
    return gt

class EntityResolutionPipeline:
    def __init__(self, val_split=0.20, random_seed=42):
        self.val_split = val_split
        self.random_seed = random_seed
        self.blocking_engine = BlockingEngine(min_token_len=3, min_ngram_overlap=2)
        self.feature_extractor = FeatureExtractor()
        self.matching_model = EntityMatchingModel(random_state=random_seed)
        self.validation_results = {}

    def run(self, dataset_dir="dataset", output_dir="output"):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        os.makedirs(output_dir, exist_ok=True)

        print("==================================================")
        print("  Starting Business Entity Resolution Pipeline")
        print("==================================================")

        # 1. Load Training Data
        train_s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
        train_s2_path = os.path.join(dataset_dir, "train", "train_source2.tsv")
        train_s3_path = os.path.join(dataset_dir, "train", "train_source3.tsv")
        train_gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

        print("\n[Step 1] Loading and preprocessing training datasets...")
        raw_train_s1 = read_tsv_dict(train_s1_path)
        raw_train_s2 = read_tsv_dict(train_s2_path)
        raw_train_s3 = read_tsv_dict(train_s3_path)
        train_gt = read_ground_truth(train_gt_path)

        train_s1 = [preprocess_record(r) for r in raw_train_s1]
        train_s2 = [preprocess_record(r) for r in raw_train_s2]
        train_s3 = [preprocess_record(r) for r in raw_train_s3]
        train_target = train_s2 + train_s3
        train_target_dict = {r['entity_id']: r for r in train_target}

        print(f"  Source 1 Records: {len(train_s1)}")
        print(f"  Source 2 Records: {len(train_s2)}")
        print(f"  Source 3 Records: {len(train_s3)}")
        print(f"  Ground Truth Links: {len(train_gt)}")

        # 2. Train / Validation Split (Split S1 entities)
        shuffled_s1 = list(train_s1)
        random.shuffle(shuffled_s1)
        n_val = max(1, int(len(shuffled_s1) * self.val_split))
        val_s1 = shuffled_s1[:n_val]
        train_sub_s1 = shuffled_s1[n_val:]
        val_s1_ids = [r['entity_id'] for r in val_s1]
        val_s1_dict = {r['entity_id']: r for r in val_s1}

        print(f"\n[Step 2] Created validation split: Train={len(train_sub_s1)} entities, Val={len(val_s1)} entities.")

        # 3. Fit Blocking and TF-IDF on Training Data
        print("\n[Step 3] Fitting Blocking Engine and TF-IDF feature extractors...")
        self.blocking_engine.fit(train_target)
        self.feature_extractor.fit(train_s1 + train_target)

        # 4. Generate Candidates for Train & Validation
        print("\n[Step 4] Generating candidate pairs...")
        val_candidates = self.blocking_engine.generate_all_candidates(val_s1, max_candidates=50)

        # Blocking Metrics on Validation Split
        val_gt_subset = {r['entity_id']: train_gt.get(r['entity_id'], []) for r in val_s1}
        blocking_metrics = self.blocking_engine.compute_recall_and_reduction(
            val_candidates, val_gt_subset, len(train_target)
        )
        print(f"  Validation Blocking Recall Ceiling: {blocking_metrics['recall_ceiling']*100:.2f}%")
        print(f"  Validation Reduction Ratio:        {blocking_metrics['reduction_ratio']*100:.4f}%")
        print(f"  Avg Candidates per S1 Entity:      {blocking_metrics['avg_candidates_per_s1']:.1f}")

        # Candidates for Training Split
        train_candidates = self.blocking_engine.generate_all_candidates(train_sub_s1, max_candidates=50)

        # 5. Build Training Feature Matrix
        print("\n[Step 5] Building training pairs and feature matrix...")
        train_pairs = []
        train_labels = []

        for r1 in train_sub_s1:
            s1_id = r1['entity_id']
            true_matches = set(train_gt.get(s1_id, []))
            cands = train_candidates.get(s1_id, [])

            # Add negative & positive candidate pairs
            for t_id in cands:
                t_rec = train_target_dict.get(t_id)
                if t_rec:
                    is_match = 1.0 if t_id in true_matches else 0.0
                    train_pairs.append((r1, t_rec))
                    train_labels.append(is_match)

            # Ensure all ground-truth matches are included in training (high recall learning)
            for t_id in true_matches:
                if t_id not in cands:
                    t_rec = train_target_dict.get(t_id)
                    if t_rec:
                        train_pairs.append((r1, t_rec))
                        train_labels.append(1.0)

        X_train = self.feature_extractor.extract_matrix(train_pairs)
        y_train = np.array(train_labels, dtype=np.float32)

        pos_count = int(np.sum(y_train))
        neg_count = len(y_train) - pos_count
        print(f"  Training feature matrix: {X_train.shape[0]} pairs ({pos_count} positive, {neg_count} negative)")

        # 6. Train Matching Model
        print("\n[Step 6] Training Matching Classifier...")
        self.matching_model.train(X_train, y_train)

        # 7. Optimize Threshold on Validation Split
        print("\n[Step 7] Tuning classification threshold for Macro F0.5...")
        # Precompute val features
        val_pairs_cache = {}
        for s1_rec in val_s1:
            s1_id = s1_rec['entity_id']
            for t_id in val_candidates.get(s1_id, []):
                t_rec = train_target_dict.get(t_id)
                if t_rec:
                    feat = self.feature_extractor.extract_pair_features(s1_rec, t_rec)
                    val_pairs_cache[(s1_id, t_id)] = feat

        optimal_tau = self.matching_model.optimize_threshold(
            val_s1_ids, val_candidates, val_pairs_cache, val_gt_subset, self.feature_extractor
        )

        # Final Validation Evaluation
        val_preds = {}
        for s1_id in val_s1_ids:
            matches = []
            for t_id in val_candidates.get(s1_id, []):
                feat = val_pairs_cache.get((s1_id, t_id))
                if feat:
                    prob = float(self.matching_model.predict_proba(np.array([feat], dtype=np.float32))[0])
                    if prob >= optimal_tau:
                        matches.append(t_id)
            val_preds[s1_id] = matches

        val_eval = compute_macro_f05(val_gt_subset, val_preds)
        self.validation_results = {
            'blocking_metrics': blocking_metrics,
            'optimal_threshold': optimal_tau,
            'val_macro_f05': val_eval['macro_f05'],
            'val_macro_precision': val_eval['macro_precision'],
            'val_macro_recall': val_eval['macro_recall'],
            'num_val_entities': len(val_s1_ids)
        }

        print("\n=== Validation Results ===")
        print(f"  Macro F0.5:      {val_eval['macro_f05']:.4f}")
        print(f"  Macro Precision: {val_eval['macro_precision']:.4f}")
        print(f"  Macro Recall:    {val_eval['macro_recall']:.4f}")

        # 8. Run Complete Pipeline on TEST DATA
        print("\n[Step 8] Running pipeline on TEST DATA...")
        test_s1_path = os.path.join(dataset_dir, "test", "test_source1.tsv")
        test_s2_path = os.path.join(dataset_dir, "test", "test_source2.tsv")
        test_s3_path = os.path.join(dataset_dir, "test", "test_source3.tsv")

        raw_test_s1 = read_tsv_dict(test_s1_path)
        raw_test_s2 = read_tsv_dict(test_s2_path)
        raw_test_s3 = read_tsv_dict(test_s3_path)

        test_s1 = [preprocess_record(r) for r in raw_test_s1]
        test_s2 = [preprocess_record(r) for r in raw_test_s2]
        test_s3 = [preprocess_record(r) for r in raw_test_s3]
        test_target = test_s2 + test_s3
        test_target_dict = {r['entity_id']: r for r in test_target}

        print(f"  Test Source 1 Records: {len(test_s1)}")
        print(f"  Test Source 2 Records: {len(test_s2)}")
        print(f"  Test Source 3 Records: {len(test_s3)}")

        # Index test targets into blocking engine
        self.blocking_engine.fit(test_target)
        test_candidates = self.blocking_engine.generate_all_candidates(test_s1, max_candidates=50)

        # 9. Write output/candidate_pairs.tsv
        cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
        with open(cand_out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for rec in test_s1:
                s1_id = rec['entity_id']
                cands = test_candidates.get(s1_id, [])
                # Ensure sorted, no duplicates
                unique_cands = sorted(list(dict.fromkeys(cands)))
                writer.writerow([s1_id, ",".join(unique_cands)])
        print(f"\n[Step 9] Saved candidate pairs: {cand_out_path}")

        # 10. Model Scoring & Write output/matching_results.tsv
        match_out_path = os.path.join(output_dir, "matching_results.tsv")
        test_matches_map = {}

        for rec in test_s1:
            s1_id = rec['entity_id']
            cands = test_candidates.get(s1_id, [])
            matched = []
            for t_id in cands:
                t_rec = test_target_dict.get(t_id)
                if t_rec:
                    feat = self.feature_extractor.extract_pair_features(rec, t_rec)
                    prob = float(self.matching_model.predict_proba(np.array([feat], dtype=np.float32))[0])
                    if prob >= optimal_tau:
                        matched.append(t_id)
            # Ensure every match is valid, unique, and strictly within candidates
            unique_matched = sorted(list(dict.fromkeys(matched)))
            test_matches_map[s1_id] = unique_matched

        with open(match_out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for rec in test_s1:
                s1_id = rec['entity_id']
                matches = test_matches_map.get(s1_id, [])
                writer.writerow([s1_id, ",".join(matches)])
        print(f"[Step 10] Saved final matches: {match_out_path}")

        print("\n==================================================")
        print("  Pipeline Execution Complete!")
        print("==================================================")
        return self.validation_results
