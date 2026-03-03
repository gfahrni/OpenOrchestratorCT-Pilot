# emphysema.py
# Runner for Emphysema Quantification.
# Follows the exact same pattern as totalsegmentator.py:
#   - NIfTI → temp DICOM series
#   - writes a small temp Python script into work_dir
#   - calls it via: conda run -n emphysema python <tmp_script> ...
#   - reads the JSON result
#   - cleans up
# Nothing touches the original module files.

import json
import os
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

from orchestrator.runners.base import BaseRunner


LAA_THRESHOLD_PCT = 10.0


class EmphysemaRunner(BaseRunner):

    def __init__(self, config: dict):
        super().__init__(config)
        self.conda_env    = config.get("conda_env", "emphysema")
        self.threshold_hu = config.get("threshold_hu", -950)
        self.module_dir   = Path(
            config.get("module_dir") or config.get("script_path", "modules/Emphysema_Quantification")
        ).resolve()

    # ------------------------------------------------------------------
    # check_installation
    # ------------------------------------------------------------------

    def check_installation(self) -> bool:
        """Check that the module exists AND the conda env is reachable."""
        if not (self.module_dir / "data_processing_utils.py").exists():
            return False
        try:
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "python", "--version"],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, nifti_path: str, work_dir: str) -> dict:
        dcm_tmp     = os.path.join(work_dir, "dcm_tmp_emphysema")
        tmp_script  = os.path.join(work_dir, "run_emphysema_tmp.py")
        result_json = os.path.join(work_dir, "emphysema_result.json")

        try:
            # ── Step 1 : NIfTI → DICOM ────────────────────────────────
            print("  [emphysema] Converting NIfTI → DICOM...")
            nz = self._nifti_to_dicom(nifti_path, dcm_tmp)
            print(f"  [emphysema] {nz} slices written to {dcm_tmp}")

            # ── Step 2 : Write temp script ─────────────────────────────
            self._write_tmp_script(tmp_script, dcm_tmp, result_json)

            # ── Step 3 : conda run ─────────────────────────────────────
            print(f"  [emphysema] Running via conda env '{self.conda_env}'...")
            cmd = [
                "conda", "run", "--no-capture-output",
                "-n", self.conda_env,
                "python", tmp_script,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800,
            )

            if result.returncode != 0:
                return {
                    "status": "failed",
                    "error": result.stderr.strip() or "emphysema script returned non-zero exit code",
                    "findings": [],
                }

            # ── Step 4 : Read JSON result ──────────────────────────────
            if not os.path.exists(result_json):
                return {
                    "status": "failed",
                    "error": "emphysema_result.json not found after script run",
                    "findings": [],
                }

            with open(result_json) as f:
                data = json.load(f)

            if data.get("status") == "failed":
                return {"status": "failed", "error": data.get("error"), "findings": []}

            return self._interpret_results(data["score_3d"], data["score_2d"])

        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "Emphysema script timed out", "findings": []}
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "findings": []}

        finally:
            # Delete dcm_tmp early to free disk space during the run
            if os.path.exists(dcm_tmp):
                shutil.rmtree(dcm_tmp)
                print(f"  [emphysema] Temp DICOM folder deleted.")

    # ------------------------------------------------------------------
    # _write_tmp_script
    # ------------------------------------------------------------------

    def _write_tmp_script(self, script_path: str, dicom_dir: str, output_json: str):
        """
        Write a self-contained Python script into work_dir.
        Runs inside the emphysema conda env and writes a JSON result.
        """
        script = textwrap.dedent(f"""\
            import sys, json
            sys.path.insert(0, r"{self.module_dir}")

            from data_processing_utils import wrapper_fn

            result = wrapper_fn(r"{dicom_dir}", thres={self.threshold_hu})

            if result is None:
                out = {{"status": "failed", "error": "wrapper_fn returned None"}}
            else:
                _, _, _, _, score_3d, score_2d = result
                out = {{"status": "success", "score_3d": float(score_3d), "score_2d": float(score_2d)}}

            with open(r"{output_json}", "w") as f:
                json.dump(out, f)

            print(f"score_3d={{score_3d:.4f}}  score_2d={{score_2d:.4f}}")
        """)

        with open(script_path, "w") as f:
            f.write(script)

    # ------------------------------------------------------------------
    # _nifti_to_dicom
    # ------------------------------------------------------------------

    def _nifti_to_dicom(self, nifti_path: str, output_dir: str) -> int:
        os.makedirs(output_dir, exist_ok=True)

        img = nib.load(nifti_path)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()
        spacing_x, spacing_y, spacing_z = img.header.get_zooms()

        data = np.transpose(data, (2, 1, 0))
        data = np.flip(data, axis=1)
        data = np.flip(data, axis=2)
        data = np.round(data).astype(np.int16)
        nz, ny, nx = data.shape

        study_uid  = generate_uid()
        series_uid = generate_uid()
        frame_uid  = generate_uid()

        for i in range(nz):
            filename = os.path.join(output_dir, f"slice_{i:04d}.dcm")
            ds = FileDataset(filename, {}, file_meta=Dataset(), preamble=b"\0" * 128)
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds.SOPClassUID               = pydicom.uid.CTImageStorage
            ds.SOPInstanceUID            = generate_uid()
            ds.StudyInstanceUID          = study_uid
            ds.SeriesInstanceUID         = series_uid
            ds.FrameOfReferenceUID       = frame_uid
            ds.Modality                  = "CT"
            ds.PatientName               = "Emphysema^Runner"
            ds.PatientID                 = "0001"
            ds.StudyDate                 = datetime.now().strftime('%Y%m%d')
            ds.StudyTime                 = datetime.now().strftime('%H%M%S')
            ds.SeriesNumber              = 1
            ds.InstanceNumber            = i + 1
            ds.Rows                      = ny
            ds.Columns                   = nx
            ds.PixelSpacing              = [str(spacing_y), str(spacing_x)]
            ds.SliceThickness            = str(spacing_z)
            ds.ImageOrientationPatient   = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient      = [0.0, 0.0, float(i * spacing_z)]
            ds.WindowCenter              = "-600"
            ds.WindowWidth               = "1500"
            ds.RescaleIntercept          = "0"
            ds.RescaleSlope              = "1"
            ds.SamplesPerPixel           = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsStored                = 16
            ds.BitsAllocated             = 16
            ds.HighBit                   = 15
            ds.PixelRepresentation       = 1
            ds.PixelData                 = data[i].tobytes()

            ds.save_as(filename)

        return nz

    # ------------------------------------------------------------------
    # _interpret_results
    # ------------------------------------------------------------------

    def _interpret_results(self, score_3d: float, score_2d: float) -> dict:
        pct_3d   = round(score_3d * 100, 2)
        pct_2d   = round(score_2d * 100, 2)
        severity = self._classify(pct_3d)
        flagged  = pct_3d >= LAA_THRESHOLD_PCT

        flagged_findings = []
        if flagged:
            flagged_findings.append({
                "finding":   "Significant emphysema",
                "laa_pct":   pct_3d,
                "threshold": LAA_THRESHOLD_PCT,
                "severity":  severity,
            })

        return {
            "status":           "success",
            "error":            None,
            "laa_950_percent":  pct_3d,
            "laa_950_pct_2d":   pct_2d,
            "severity":         severity,
            "flagged_findings": flagged_findings,
            "hu_threshold":     self.threshold_hu,
        }

    @staticmethod
    def _classify(laa_pct: float) -> str:
        if laa_pct < 2.7:
            return "none"
        elif laa_pct < 8.5:
            return "mild to moderate"
        else:
            return "severe"