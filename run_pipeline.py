#!/usr/bin/env python3
"""
Main execution script for Business Entity Resolution Challenge.
Runs the complete training and inference pipeline, generates the required outputs,
and executes the official validator.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution"))

# If sklearn not in default path, add AquaCycleML venv site-packages if present
venv_site = r"C:\Users\jegan\Downloads\AquaCycleML (1)\AquaCycleML\.venv\Lib\site-packages"
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.append(venv_site)

from src.pipeline import EntityResolutionPipeline
from utils.validate_submission import validate

def main():
    dataset_dir = "dataset"
    output_dir = "output"

    pipeline = EntityResolutionPipeline(val_split=0.20, random_seed=42)
    val_results = pipeline.run(dataset_dir=dataset_dir, output_dir=output_dir)

    # Execute official validator
    print("\n[Step 11] Running Official Submission Validator...")
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    test_dir = os.path.join(dataset_dir, "test")

    is_valid = validate(matching_file, candidate_file, test_dir)
    if is_valid:
        print("\nAll validation checks PASSED successfully! (exit 0)")
        sys.exit(0)
    else:
        print("\nValidation FAILED! Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
