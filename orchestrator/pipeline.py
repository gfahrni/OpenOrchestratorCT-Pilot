# pipeline.py
# Orchestrates the analysis pipeline.
# Reads config, instantiates enabled runners, checks installations,
# runs each module, and returns all findings for report generation.

import os
import tempfile
from pathlib import Path

import yaml

from orchestrator.runners.totalsegmentator import TotalSegmentatorRunner
from orchestrator.runners.emphysema import EmphysemaRunner
from orchestrator.runners.nodule import NoduleRunner


def load_config(config_path: str) -> dict:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Parsed config as a dict
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(nifti_path: str, config_path: str = "config.yaml") -> dict:
    """
    Main pipeline entry point.

    For each enabled module:
      1. Instantiate the runner
      2. Check the tool is installed
      3. Run the analysis
      4. Collect findings

    Args:
        nifti_path:  Absolute path to the input .nii.gz file
        config_path: Path to config.yaml (default: config.yaml in project root)

    Returns:
        A results dict keyed by module name, e.g.:
        {
            "input": "/path/to/CT_PE_017.nii.gz",
            "totalsegmentator": { "status": "success", "volumes_ml": {...}, ... },
            "emphysema":        { "status": "skipped" },
            "lung_nodule":      { "status": "skipped" },
        }
    """
    config = load_config(config_path)
    modules_config = config.get("modules", {})

    # Validate input file exists
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Input NIfTI file not found: {nifti_path}")

    # Map module names to their runner classes
    runner_registry = {
        "totalsegmentator": TotalSegmentatorRunner,
        "emphysema":        EmphysemaRunner,
        "lung_nodule":      NoduleRunner,
    }

    results = {
        "input": os.path.abspath(nifti_path),
    }

    # Create a shared temporary working directory for all modules
    # Each runner will create its own subdirectory inside it
    with tempfile.TemporaryDirectory(prefix="thoracic_ct_") as work_dir:
        print(f"Working directory: {work_dir}")

        for module_name, RunnerClass in runner_registry.items():
            module_config = modules_config.get(module_name, {})

            # Skip if module is disabled in config
            if not module_config.get("enabled", False):
                print(f"[{module_name}] Skipped (disabled in config)")
                results[module_name] = {"status": "skipped"}
                continue

            print(f"\n[{module_name}] Starting...")
            runner = RunnerClass(module_config)

            # Verify the external tool is reachable before attempting to run
            print(f"[{module_name}] Checking installation...")
            if not runner.check_installation():
                print(f"[{module_name}] Installation check FAILED — skipping")
                results[module_name] = {
                    "status": "failed",
                    "error": (
                        f"Tool not found. Make sure the conda environment "
                        f"'{module_config.get('conda_env', '?')}' is created "
                        f"and the tool is installed inside it."
                    ),
                    "findings": [],
                }
                continue

            print(f"[{module_name}] Installation OK — running analysis...")
            findings = runner.run(
                nifti_path=os.path.abspath(nifti_path),
                work_dir=work_dir,
            )

            results[module_name] = findings

            # Print a brief status summary
            status = findings.get("status", "unknown")
            if status == "success":
                n_flagged = len(findings.get("flagged_findings", []))
                print(f"[{module_name}] Done — {n_flagged} finding(s) flagged")
            else:
                print(f"[{module_name}] Failed: {findings.get('error', 'unknown error')}")

    return results