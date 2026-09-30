#!/usr/bin/env python3
"""
Standalone Runner for Business Entity Resolution Pipeline.
Allows evaluating and reproducing the pipeline from inside the code/business_entity_resolution directory.
"""
import argparse
import sys
import os

# Ensure package root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Fallback for scikit-learn in local environment if needed
local_venv = r"C:\Users\jegan\Downloads\AquaCycleML (1)\AquaCycleML\.venv\Lib\site-packages"
if os.path.exists(local_venv) and local_venv not in sys.path:
    sys.path.append(local_venv)

from src.pipeline import EntityResolutionPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--dataset-dir", default=None, help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output-dir", default=None, help="Path to output directory for TSV results")
    parser.add_argument("--val-split", type=float, default=0.20, help="Held-out validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Determine default paths relative to workspace or current dir
    if args.dataset_dir is None:
        if os.path.exists("dataset"):
            dataset_dir = "dataset"
        elif os.path.exists("../../dataset"):
            dataset_dir = "../../dataset"
        elif os.path.exists("../dataset"):
            dataset_dir = "../dataset"
        else:
            dataset_dir = "dataset"
    else:
        dataset_dir = args.dataset_dir

    if args.output_dir is None:
        if os.path.exists("output") or not os.path.exists("../../output"):
            output_dir = "output"
        else:
            output_dir = "../../output"
    else:
        output_dir = args.output_dir

    pipeline = EntityResolutionPipeline(val_split=args.val_split, random_seed=args.seed)
    results = pipeline.run(dataset_dir=dataset_dir, output_dir=output_dir)

    print("\nStandalone pipeline run completed successfully.")
    print(f"Validation Macro F0.5: {results.get('val_macro_f05', 0.0):.4f}")

if __name__ == "__main__":
    main()
