"""
Matching Model & F0.5 Evaluation Module.
Trains a precision-optimized classifier to score candidate pairs.
Implements the exact macro-averaged F0.5 evaluation metric with singleton handling.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def compute_entity_f05(true_ids: set, pred_ids: set) -> float:
    """
    Computes F0.5 for a single Source 1 entity according to the challenge rules:
    - If true is empty (singleton):
        scores 1.0 if pred is empty, else 0.0
    - If true is non-empty:
        scores 0.0 if pred is empty
        else (1.25 * P * R) / (0.25 * P + R)
    """
    if not true_ids:
        # Ground truth is singleton (no matches)
        return 1.0 if not pred_ids else 0.0

    if not pred_ids:
        # Ground truth has matches, but model predicted none
        return 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom

def compute_macro_f05(ground_truth_map: dict[str, list[str]], prediction_map: dict[str, list[str]]) -> dict:
    """
    Computes macro-averaged F0.5 across all Source 1 entities in the evaluation set.
    """
    scores = []
    entity_metrics = []

    for s1_id, true_list in ground_truth_map.items():
        true_set = set(true_list)
        pred_set = set(prediction_map.get(s1_id, []))
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)

        is_singleton = (len(true_set) == 0)
        tp = len(true_set & pred_set)
        p = (tp / len(pred_set)) if pred_set else (1.0 if is_singleton else 0.0)
        r = (tp / len(true_set)) if true_set else (1.0 if not pred_set else 0.0)

        entity_metrics.append({
            's1_id': s1_id,
            'f05': score,
            'precision': p,
            'recall': r,
            'is_singleton': is_singleton,
            'num_true': len(true_set),
            'num_pred': len(pred_set)
        })

    macro_f05 = float(np.mean(scores)) if scores else 0.0

    # Macro precision and recall for reporting
    avg_p = float(np.mean([m['precision'] for m in entity_metrics]))
    avg_r = float(np.mean([m['recall'] for m in entity_metrics]))

    return {
        'macro_f05': macro_f05,
        'macro_precision': avg_p,
        'macro_recall': avg_r,
        'total_entities': len(ground_truth_map),
        'entity_scores': entity_metrics
    }

class EntityMatchingModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.optimal_threshold = 0.50
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on positive and negative feature vectors."""
        if len(y) == 0:
            raise ValueError("Cannot train on empty dataset")
        self.model.fit(X, y)
        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns match probabilities for pairs."""
        if not self.is_trained or len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        else:
            return probs[:, 0]

    def optimize_threshold(self, s1_val_ids: list[str], val_candidates: dict[str, list[str]],
                           val_pairs_X: dict[tuple[str, str], list[float]],
                           val_gt: dict[str, list[str]],
                           feature_extractor) -> float:
        """
        Sweeps decision threshold tau to maximize Macro F0.5 on held-out validation set.
        Prioritizes precision to align with F0.5 weighting.
        """
        # Score all candidate pairs once
        pair_keys = []
        pair_rows = []
        for s1_id in s1_val_ids:
            for t_id in val_candidates.get(s1_id, []):
                feat = val_pairs_X.get((s1_id, t_id))
                if feat is not None:
                    pair_keys.append((s1_id, t_id))
                    pair_rows.append(feat)

        if not pair_rows:
            self.optimal_threshold = 0.55
            return self.optimal_threshold

        X_val = np.array(pair_rows, dtype=np.float32)
        probs = self.predict_proba(X_val)

        prob_map = {}
        for (s1_id, t_id), p in zip(pair_keys, probs):
            prob_map[(s1_id, t_id)] = p

        best_tau = 0.50
        best_score = -1.0
        best_details = {}

        # Search thresholds between 0.30 and 0.85
        threshold_candidates = np.linspace(0.30, 0.85, 23)
        for tau in threshold_candidates:
            pred_map = {}
            for s1_id in s1_val_ids:
                matches = []
                for t_id in val_candidates.get(s1_id, []):
                    p = prob_map.get((s1_id, t_id), 0.0)
                    if p >= tau:
                        matches.append(t_id)
                pred_map[s1_id] = matches

            val_metrics = compute_macro_f05({s1_id: val_gt.get(s1_id, []) for s1_id in s1_val_ids}, pred_map)
            score = val_metrics['macro_f05']

            if score > best_score:
                best_score = score
                best_tau = tau
                best_details = val_metrics

        self.optimal_threshold = float(best_tau)
        print(f"Optimal Threshold Selected: {best_tau:.3f} (Macro F0.5: {best_score:.4f}, Precision: {best_details.get('macro_precision', 0):.4f}, Recall: {best_details.get('macro_recall', 0):.4f})")
        return self.optimal_threshold
