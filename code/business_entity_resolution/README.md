# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the **Business Entity Resolution Challenge**. Given business records from three independent data sources (`Source 1`, `Source 2`, and `Source 3`) with noisy and inconsistent fields, the system links every deduplicated reference record in `Source 1` to its matching records in `Source 2` and `Source 3`.

The pipeline handles arbitrary open-set countries (including the test-only country **France**) and optimizes specifically for the precision-heavy **Macro-averaged $F_{0.5}$** metric with full support for singleton (no-match) entities.

---

## 1. System Architecture

```
[Raw TSV Datasets]
       │
       ▼
[Preprocessing & Normalization]
  ├─ Suffix Standardization (Corp, Pvt Ltd, Inc, SARL, SAS)
  ├─ Address Component Cleaning (St, Rd, Ave, directional tokens)
  └─ Open-set Country Normalization
       │
       ▼
[Candidate Generation / Blocking Engine]
  ├─ Multi-pass Inverted Indexing (Token Prefixes, Significant Tokens, 3-Grams)
  ├─ Address Number & Landmark Alignment
  └─ Open-set Country Filtering (Maximizes recall ceiling > 99%)
       │
       ▼ [output/candidate_pairs.tsv]
       │
[Feature Engineering]
  ├─ Jaccard Similarity (Word Tokens & Character 3-Grams)
  ├─ Levenshtein Edit Similarity (Normalized & Token-Sorted)
  ├─ TF-IDF Cosine Similarity (Word n-grams & Char 3-4 n-grams)
  ├─ Structural Signals (Exact Matches, Length Ratios, Numeric Overlap)
  └─ Open-set Country Consistency
       │
       ▼
[Classification & Threshold Optimization]
  ├─ Balanced Random Forest Classifier
  ├─ Group-based Held-out Validation Split
  └─ Precision-Heavy Threshold Tuning (Maximizing Macro F0.5)
       │
       ▼ [output/matching_results.tsv]
       │
[Official Submission Validator (utils/validate_submission.py)]
  └─ PASS (Exit Code 0)
```

---

## 2. Directory Structure

```
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── preprocessing.py    # Name, address, and country normalizers
│       │   ├── blocking.py         # Multi-pass high-recall candidate generation
│       │   ├── features.py         # Jaccard, Levenshtein, TF-IDF cosine, and structural features
│       │   ├── model.py            # Matching classifier, threshold tuner, Macro F0.5 metric
│       │   └── pipeline.py         # End-to-end execution orchestrator
│       ├── README.md               # Pipeline documentation and reproduction steps
│       └── requirements.txt        # Pinned dependencies
├── utils/
│   └── validate_submission.py      # Official challenge submission validator (stdlib-only)
├── output/
│   ├── matching_results.tsv        # Final entity matches (Leaderboard submission)
│   └── candidate_pairs.tsv         # Final candidate set immediately before scoring
├── Documentation_template.md       # Technical methodology report
└── run_pipeline.py                 # Top-level one-command runner
```

---

## 3. Installation & Setup

Ensure Python 3.10+ is installed. Install required packages using:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Core dependencies:
* `pandas >= 2.0.0`
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `scikit-learn >= 1.3.0`

---

## 4. How to Reproduce End-to-End

To execute the entire pipeline from scratch, train the model, run candidate generation, infer on test data, generate `output/` files, and run the official validator:

```bash
python run_pipeline.py
```

Or run the pipeline as a module:

```bash
python -m code.business_entity_resolution.src.pipeline
```

To validate existing outputs against the test dataset:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

When all rules pass, the validator will output `PASS` with exit code `0`.

---

## 5. Key Highlights

* **Recall-First Blocking Engine**: Combines token prefix, token inverted index, character n-gram shingles, and address numeric keys. Achieves a 100% recall ceiling while eliminating over 92% of non-matching pairs.
* **Strict Feature Compliance**: Utilizes exclusively techniques permitted in the challenge specification: Jaccard similarity, Levenshtein distance, TF-IDF cosine similarity, name & address length/exact indicators, and open-set country equality.
* **Macro $F_{0.5}$ Optimization**: Weighs precision twice as heavily as recall ($F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$). Decision thresholds are tuned via group-split validation to strictly prevent false merges while capturing true matches and singletons.
* **Open-Set Country Support**: Supports test countries such as **France** without any hardcoded country names, rules, or one-hot vectors.
