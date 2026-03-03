# run.py
# Main entrypoint for the Thoracic CT Orchestrator.
# Usage: python run.py --input path/to/ct.nii.gz

import argparse
import os
import sys

from orchestrator.pipeline import run_pipeline
from orchestrator.report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thoracic CT Orchestrator — runs AI analysis modules on a NIfTI CT scan."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input NIfTI file (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Override output directory from config.yaml"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  Thoracic CT Orchestrator")
    print("=" * 60)
    print(f"  Input : {args.input}")
    print(f"  Config: {args.config}")
    print("=" * 60)

    # Run the pipeline — returns a dict of findings per module
    results = run_pipeline(
        nifti_path=args.input,
        config_path=args.config,
    )

    # Generate the .txt report
    report_path = generate_report(
        results=results,
        input_path=args.input,
        config_path=args.config,
        output_dir_override=args.output_dir,
    )

    print("\n" + "=" * 60)
    print(f"  Report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()