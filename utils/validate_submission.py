#!/usr/bin/env python3
"""
Official Submission Validator for Business Entity Resolution Challenge.
Stdlib-only, no external dependencies.
Checks matching_results.tsv and candidate_pairs.tsv against all competition rules.
"""
import argparse
import csv
import os
import sys

def validate(matching_file, candidate_file, test_dir):
    issues = []

    # 1. Check file existence
    if not os.path.exists(matching_file):
        issues.append(f"Matching file not found: {matching_file}")
    if not os.path.exists(candidate_file):
        issues.append(f"Candidate file not found: {candidate_file}")

    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    s2_test_path = os.path.join(test_dir, "test_source2.tsv")
    s3_test_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_test_path, s2_test_path, s3_test_path]:
        if not os.path.exists(p):
            issues.append(f"Test source file not found: {p}")

    if issues:
        print_report(issues)
        return False

    # 2. Load ground valid test IDs
    def load_ids(filepath):
        ids = set()
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader, None)
            if not headers or "entity_id" not in headers:
                issues.append(f"Missing 'entity_id' column in {filepath}")
                return ids
            id_idx = headers.index("entity_id")
            for line_no, row in enumerate(reader, start=2):
                if row:
                    ids.add(row[id_idx].strip())
        return ids

    valid_s1_ids = load_ids(s1_test_path)
    valid_s2_ids = load_ids(s2_test_path)
    valid_s3_ids = load_ids(s3_test_path)
    valid_target_ids = valid_s2_ids | valid_s3_ids

    # 3. Validate matching_results.tsv
    matching_map = {}
    with open(matching_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader, None)
        if not headers or headers != ["source1_entity_id", "matched_entity_ids"]:
            issues.append(f"Invalid headers in {matching_file}. Expected ['source1_entity_id', 'matched_entity_ids'], got {headers}")
        else:
            for line_no, row in enumerate(reader, start=2):
                if len(row) < 1:
                    continue
                s1_id = row[0].strip()
                matched_str = row[1].strip() if len(row) > 1 else ""

                if s1_id in matching_map:
                    issues.append(f"Duplicate source1_entity_id '{s1_id}' in {matching_file} at line {line_no}")
                if s1_id not in valid_s1_ids:
                    issues.append(f"Invalid source1_entity_id '{s1_id}' at line {line_no} not found in {s1_test_path}")

                matched_ids = [x.strip() for x in matched_str.split(",") if x.strip()]
                # Check duplicate matched IDs
                if len(matched_ids) != len(set(matched_ids)):
                    issues.append(f"Duplicate IDs in matched_entity_ids for '{s1_id}' at line {line_no}: {matched_str}")
                # Check valid S2/S3 IDs
                for m_id in matched_ids:
                    if m_id.startswith("S1-"):
                        issues.append(f"Illegal self-match '{m_id}' in matched_entity_ids for '{s1_id}' at line {line_no}")
                    elif m_id not in valid_target_ids:
                        issues.append(f"Invalid matched entity ID '{m_id}' for '{s1_id}' at line {line_no} (not in Source 2 or Source 3 test set)")

                matching_map[s1_id] = set(matched_ids)

    # 4. Check all test S1 IDs appear in matching_results.tsv
    missing_in_matching = valid_s1_ids - set(matching_map.keys())
    if missing_in_matching:
        issues.append(f"{len(missing_in_matching)} Source 1 test entities missing from {matching_file}. Example: {list(missing_in_matching)[:5]}")

    # 5. Validate candidate_pairs.tsv
    candidate_map = {}
    with open(candidate_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader, None)
        if not headers or headers != ["source1_entity_id", "candidate_entity_ids"]:
            issues.append(f"Invalid headers in {candidate_file}. Expected ['source1_entity_id', 'candidate_entity_ids'], got {headers}")
        else:
            for line_no, row in enumerate(reader, start=2):
                if len(row) < 1:
                    continue
                s1_id = row[0].strip()
                cand_str = row[1].strip() if len(row) > 1 else ""

                if s1_id in candidate_map:
                    issues.append(f"Duplicate source1_entity_id '{s1_id}' in {candidate_file} at line {line_no}")
                if s1_id not in valid_s1_ids:
                    issues.append(f"Invalid source1_entity_id '{s1_id}' at line {line_no} not found in {s1_test_path}")

                cand_ids = [x.strip() for x in cand_str.split(",") if x.strip()]
                # Check duplicate candidate IDs
                if len(cand_ids) != len(set(cand_ids)):
                    issues.append(f"Duplicate IDs in candidate_entity_ids for '{s1_id}' at line {line_no}: {cand_str}")
                # Check valid S2/S3 IDs
                for c_id in cand_ids:
                    if c_id.startswith("S1-"):
                        issues.append(f"Illegal Source 1 ID '{c_id}' in candidate_entity_ids for '{s1_id}' at line {line_no}")
                    elif c_id not in valid_target_ids:
                        issues.append(f"Invalid candidate entity ID '{c_id}' for '{s1_id}' at line {line_no} (not in Source 2 or Source 3 test set)")

                candidate_map[s1_id] = set(cand_ids)

    # 6. Check all test S1 IDs appear in candidate_pairs.tsv
    missing_in_candidates = valid_s1_ids - set(candidate_map.keys())
    if missing_in_candidates:
        issues.append(f"{len(missing_in_candidates)} Source 1 test entities missing from {candidate_file}. Example: {list(missing_in_candidates)[:5]}")

    # 7. Check that every matched entity ID appeared in candidates
    for s1_id, matched_set in matching_map.items():
        cand_set = candidate_map.get(s1_id, set())
        uncovered = matched_set - cand_set
        if uncovered:
            issues.append(f"Pipeline violation for '{s1_id}': matched IDs {uncovered} were never included in candidate_pairs.tsv")

    print_report(issues)
    return len(issues) == 0

def print_report(issues):
    if not issues:
        print("PASS")
    else:
        print(f"FAILED: Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")

def main():
    parser = argparse.ArgumentParser(description="Validate submission files for Business Entity Resolution Challenge")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")

    args = parser.parse_args()
    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
