# totalsegmentator.py
# Runner for TotalSegmentator.
# Calls TotalSegmentator via subprocess (conda run), parses statistics.json,
# applies normative volume thresholds, and returns structured findings.

import json
import os
import subprocess
import tempfile
from pathlib import Path

from orchestrator.runners.base import BaseRunner


# ---------------------------------------------------------------------------
# Normative thresholds (volumes in mL)
# Source: Remark et al., Radiology AI 2025
# https://pubs.rsna.org/doi/10.1148/ryai.250506
#
# These are conservative upper-limit thresholds used to flag findings.
# A more complete implementation would use sex/age-adjusted percentiles.
# ---------------------------------------------------------------------------

THRESHOLDS = {
    # Cardiomegaly: heart volume > 670 mL (men) / 524 mL (women)
    # Using the higher male threshold as a conservative default when sex unknown
    "heart": {
        "max_ml": 670,
        "finding": "Cardiomegaly",
        "normal_range": "< 524 mL (women) / < 670 mL (men)",
    },
    # Goitre: thyroid volume > 25 mL
    "thyroid_gland": {
        "max_ml": 25,
        "finding": "Goitre",
        "normal_range": "< 25 mL",
    },
    # Adrenal nodule / hypertrophy: volume > 10 mL per gland
    "adrenal_gland_left": {
        "max_ml": 10,
        "finding": "Adrenal enlargement (left)",
        "normal_range": "< 10 mL",
    },
    "adrenal_gland_right": {
        "max_ml": 10,
        "finding": "Adrenal enlargement (right)",
        "normal_range": "< 10 mL",
    },
}

# Structures to segment — only what we need, keeps runtime low
ROI_SUBSET = ["heart", "thyroid_gland", "adrenal_gland_left", "adrenal_gland_right"]


class TotalSegmentatorRunner(BaseRunner):
    """
    Runner for TotalSegmentator.

    Calls TotalSegmentator in a subprocess inside its own conda environment,
    then parses the statistics.json output to extract organ volumes and
    flag abnormal findings against normative thresholds.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.conda_env = config.get("conda_env", "totalseg")
        # Use MPS on Apple Silicon, GPU on NVIDIA, fallback to CPU
        self.device = config.get("device", "cpu")

    def check_installation(self) -> bool:
        """
        Verify TotalSegmentator is installed in the expected conda environment
        by running: conda run -n <env> TotalSegmentator --version
        """
        try:
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "TotalSegmentator", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(self, nifti_path: str, work_dir: str) -> dict:
        """
        Run TotalSegmentator on the input NIfTI file.

        Steps:
          1. Call TotalSegmentator via conda run with --statistics and --roi_subset
          2. Parse the output statistics.json
          3. Convert mm³ → mL and apply normative thresholds
          4. Return structured findings dict

        Args:
            nifti_path: Path to the input .nii.gz file
            work_dir:   Directory where TotalSegmentator will write its outputs

        Returns:
            findings dict with status, raw volumes, and flagged findings
        """
        # Directory where TotalSegmentator will write segmentation masks + statistics.json
        seg_output_dir = os.path.join(work_dir, "totalseg_output")
        os.makedirs(seg_output_dir, exist_ok=True)

        # Build the TotalSegmentator CLI command
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "TotalSegmentator",
            "-i", nifti_path,
            "-o", seg_output_dir,
            "--roi_subset", *ROI_SUBSET,   # only segment what we need
            "--statistics",                 # output statistics.json with volumes
            "--fast",                       # 3mm model — sufficient for volume estimation
            "--device", self.device,
            "--quiet",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max
            )

            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": result.stderr.strip() or "TotalSegmentator returned non-zero exit code",
                    "findings": [],
                }

        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "TotalSegmentator timed out after 10 minutes", "findings": []}
        except FileNotFoundError:
            return {"status": "failed", "error": "conda not found — is conda installed and on PATH?", "findings": []}

        # Parse the statistics.json output
        stats_path = os.path.join(seg_output_dir, "statistics.json")
        if not os.path.exists(stats_path):
            return {"status": "failed", "error": "statistics.json not found after TotalSegmentator run", "findings": []}

        with open(stats_path, "r") as f:
            stats = json.load(f)

        return self._interpret_results(stats)

    def _interpret_results(self, stats: dict) -> dict:
        """
        Parse raw statistics.json, convert volumes, apply thresholds.

        Args:
            stats: Parsed JSON dict from statistics.json
                   e.g. {"heart": {"volume": 512676.0, "intensity": 252.2}, ...}

        Returns:
            Structured findings dict
        """
        volumes_ml = {}
        flagged = []
        notes = []

        for structure, threshold_info in THRESHOLDS.items():
            if structure not in stats:
                notes.append(f"{structure}: not found in segmentation output")
                continue

            volume_mm3 = stats[structure].get("volume", 0.0)
            volume_ml = round(volume_mm3 / 1000.0, 1)  # mm³ → mL
            volumes_ml[structure] = volume_ml

            # Volume of 0 usually means the structure was outside the field of view
            if volume_ml == 0.0:
                notes.append(f"{structure}: volume = 0 mL — likely outside field of view or not detected")
                continue

            # Apply normative threshold
            if volume_ml > threshold_info["max_ml"]:
                flagged.append({
                    "structure": structure,
                    "finding": threshold_info["finding"],
                    "volume_ml": volume_ml,
                    "threshold_ml": threshold_info["max_ml"],
                    "normal_range": threshold_info["normal_range"],
                })

        return {
            "status": "success",
            "error": None,
            "volumes_ml": volumes_ml,
            "flagged_findings": flagged,
            "notes": notes,
        }