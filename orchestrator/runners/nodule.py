# nodule.py
# Runner for Lung Nodule Detection (rlsn/LungNoduleDetection — VitDet3D).
#
# Pipeline (mirrors EmphysemaRunner pattern exactly):
#   1. Write a self-contained temp Python script into work_dir
#   2. That script:
#        - loads the NIfTI with nibabel, extracts array + origin + spacing
#        - runs the sliding-window detect() from eval.py
#        - writes a JSON result to work_dir
#   3. Call the script via: conda run -n nodules python <tmp_script>
#   4. Parse the JSON and return structured findings
#
# Nothing touches the original module files.
# Only this file + config.yaml entries are needed.

import json
import os
import subprocess
import textwrap
from pathlib import Path

from orchestrator.runners.base import BaseRunner


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Nodules >= 6 mm are considered clinically significant (Fleischner Society)
SIGNIFICANT_SIZE_MM = 6.0

# Logit threshold used in the original eval.py  (-5 = very permissive)
# Raise to 0 for higher precision (sigmoid(0) = 50% confidence)
DEFAULT_LOGIT_THRESHOLD = 0.0


class NoduleRunner(BaseRunner):
    """
    Runner for Lung Nodule Detection (VitDet3D — rlsn/LungNoduleDetection).

    Calls the model in a subprocess inside its own conda environment,
    then parses the JSON output to return structured nodule findings.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.conda_env       = config.get("conda_env", "nodules")
        self.module_dir      = Path(
            config.get("module_dir", "modules/LungNoduleDetection")
        ).resolve()
        self.checkpoint_path = Path(
            config.get("checkpoint", "modules/LungNoduleDetection/checkpoint")
        ).resolve()
        self.logit_threshold = float(config.get("logit_threshold", DEFAULT_LOGIT_THRESHOLD))
        self.device          = config.get("device", "cpu")

    # ------------------------------------------------------------------
    # check_installation
    # ------------------------------------------------------------------

    def check_installation(self) -> bool:
        """
        Verify that:
          - eval.py and model.py exist in the module directory
          - the checkpoint directory/file exists
          - the conda environment is reachable
        """
        if not (self.module_dir / "eval.py").exists():
            print(f"  [lung_nodule] eval.py not found in {self.module_dir}")
            return False
        if not (self.module_dir / "model.py").exists():
            print(f"  [lung_nodule] model.py not found in {self.module_dir}")
            return False
        if not self.checkpoint_path.exists():
            print(f"  [lung_nodule] Checkpoint not found at {self.checkpoint_path}")
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
        tmp_script  = os.path.join(work_dir, "run_nodule_tmp.py")
        result_json = os.path.join(work_dir, "nodule_result.json")

        try:
            # ── Step 1 : Write temp script ─────────────────────────────
            self._write_tmp_script(tmp_script, nifti_path, result_json)

            # ── Step 2 : conda run ─────────────────────────────────────
            print(f"  [lung_nodule] Running via conda env '{self.conda_env}'...")
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
                    "error": result.stderr.strip() or "nodule script returned non-zero exit code",
                    "findings": [],
                }

            # ── Step 3 : Read JSON result ──────────────────────────────
            if not os.path.exists(result_json):
                return {
                    "status": "failed",
                    "error": "nodule_result.json not found after script run",
                    "findings": [],
                }

            with open(result_json) as f:
                data = json.load(f)

            if data.get("status") == "failed":
                return {"status": "failed", "error": data.get("error"), "findings": []}

            return self._interpret_results(data["candidates"])

        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "Nodule script timed out", "findings": []}
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "findings": []}

    # ------------------------------------------------------------------
    # _write_tmp_script
    # ------------------------------------------------------------------

    def _write_tmp_script(self, script_path: str, nifti_path: str, output_json: str):
        """
        Write a self-contained Python script into work_dir.
        Runs inside the nodules conda env and writes a JSON result.

        The script reproduces the detect() pipeline from eval.py
        without importing eval.py directly, to avoid any side-effects
        from the __main__ block and to control the device explicitly.

        Output JSON schema:
        {
            "status": "success",
            "candidates": [
                {
                    "coordX_mm": float,   # world coordinates (mm)
                    "coordY_mm": float,
                    "coordZ_mm": float,
                    "diameter_mm": float, # estimated from bounding box
                    "logit": float,       # raw model score
                    "probability": float  # sigmoid(logit)
                },
                ...
            ]
        }
        """
        script = textwrap.dedent(f"""\
            import sys, json, os
            import numpy as np

            # ── Add module dir to path ──────────────────────────────────
            sys.path.insert(0, r"{self.module_dir}")

            import torch
            import nibabel as nib
            from model import VitDet3D
            from dataset import sliding_window_3d
            from eval import merge_cands

            MEAN  = -775.657161489884
            STD   = 962.3208802005623
            CROP_SIZE   = [40, 128, 128]
            STRIDE_RATIO = 0.75
            BATCH_SIZE  = 32
            LOGIT_THRESH = {self.logit_threshold}
            DEVICE = "{self.device}"

            # ── Helpers (inlined from eval.py) ─────────────────────────

            def l2norm(x):
                return np.sum(x**2, axis=-1, keepdims=True) ** 0.5

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            def to_coord(bbox, origin, space):
                \"\"\"Convert normalised bbox [z1,x1,y1,z2,x2,y2] → world coord + diameter.\"\"\"
                d = len(bbox.shape)
                if d == 1:
                    bbox = np.expand_dims(bbox, 0)
                space_rev  = space[::-1]   # z,x,y order
                origin_rev = origin[::-1]
                center = (bbox[:, 3:] + bbox[:, :3]) / 2 * space_rev + origin_rev
                center = center[:, ::-1]   # back to x,y,z world order
                diam = l2norm((bbox[:, 3:] - bbox[:, :3]) * space_rev) / (3 ** 0.5)
                coord = np.concatenate([center, diam], 1)
                if d == 1:
                    coord = coord.flatten()
                return coord  # [x, y, z, diam]

            def detect(model, pixel_values, offsets, origin, space):
                candidates = []
                crop_size = np.array(CROP_SIZE)
                for i, pv_batch in enumerate(torch.split(pixel_values.to(model.device), BATCH_SIZE)):
                    img_shape = np.tile(np.array(pv_batch.shape[-3:]), 2)
                    off_batch = offsets[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].numpy()
                    off_tiled = np.tile(off_batch, 2)
                    outputs = model(pixel_values=pv_batch)
                    bbox = outputs.bbox.cpu().detach().numpy() * img_shape + off_tiled
                    coord = to_coord(bbox, origin, space)
                    logits = outputs.logits.cpu().detach().numpy()
                    candidates.append(np.concatenate([coord, logits], 1))
                candidates = np.concatenate(candidates, 0)
                candidates = merge_cands(candidates)
                candidates = candidates[candidates[:, -1] > LOGIT_THRESH]
                return candidates  # [N, 5] → x,y,z,diam,logit

            # ── Load NIfTI ─────────────────────────────────────────────
            try:
                img_nib  = nib.load(r"{nifti_path}")
                img_nib  = nib.as_closest_canonical(img_nib)
                data     = img_nib.get_fdata().astype(np.float32)

                # nibabel gives (x,y,z); model expects (z,y,x)
                data = np.transpose(data, (2, 1, 0))

                zooms    = img_nib.header.get_zooms()           # (sx, sy, sz)
                spacing  = np.array([zooms[2], zooms[1], zooms[0]])   # z,y,x spacing
                affine   = img_nib.affine
                # World origin of voxel (0,0,0) in x,y,z mm
                origin   = np.array([affine[0, 3], affine[1, 3], affine[2, 3]])

            except Exception as e:
                with open(r"{output_json}", "w") as f:
                    json.dump({{"status": "failed", "error": f"NIfTI load failed: {{e}}"}}, f)
                sys.exit(1)

            # ── Sliding window patches ─────────────────────────────────
            crop_size  = np.array(CROP_SIZE)
            stride_size = (crop_size * STRIDE_RATIO).astype(int)

            # Check CT volume is large enough for at least one crop
            if any(np.array(data.shape) < crop_size):
                with open(r"{output_json}", "w") as f:
                    json.dump({{"status": "failed",
                                "error": f"CT volume {{data.shape}} too small for crop {{CROP_SIZE}}"}}, f)
                sys.exit(1)

            offsets_np, patches = sliding_window_3d(data, crop_size, stride_size)

            # Normalise
            patches = (patches - MEAN) / STD

            pixel_values = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
            offsets_t    = torch.tensor(offsets_np, dtype=torch.int32)

            # ── Load model ─────────────────────────────────────────────
            try:
                device = DEVICE
                if device == "mps" and not torch.backends.mps.is_available():
                    print("  MPS not available, falling back to CPU")
                    device = "cpu"
                if device.startswith("gpu"):
                    device = device.replace("gpu", "cuda")
                model = VitDet3D.from_pretrained(r"{self.checkpoint_path}").eval().to(device)
            except Exception as e:
                with open(r"{output_json}", "w") as f:
                    json.dump({{"status": "failed", "error": f"Model load failed: {{e}}"}}, f)
                sys.exit(1)

            # ── Run detection ──────────────────────────────────────────
            try:
                with torch.no_grad():
                    cands = detect(model, pixel_values, offsets_t, origin, spacing)
            except Exception as e:
                with open(r"{output_json}", "w") as f:
                    json.dump({{"status": "failed", "error": f"Detection failed: {{e}}"}}, f)
                sys.exit(1)

            # ── Serialise results ──────────────────────────────────────
            results = []
            for row in cands:
                x, y, z, diam, logit = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                results.append({{
                    "coordX_mm":   round(x, 2),
                    "coordY_mm":   round(y, 2),
                    "coordZ_mm":   round(z, 2),
                    "diameter_mm": round(abs(diam), 2),
                    "logit":       round(logit, 4),
                    "probability": round(float(sigmoid(logit)), 4),
                }})

            print(f"  [lung_nodule] {{len(results)}} candidate(s) above threshold (logit > {{LOGIT_THRESH}})")

            with open(r"{output_json}", "w") as f:
                json.dump({{"status": "success", "candidates": results}}, f, indent=2)
        """)

        with open(script_path, "w") as f:
            f.write(script)

    # ------------------------------------------------------------------
    # _interpret_results
    # ------------------------------------------------------------------

    def _interpret_results(self, candidates: list) -> dict:
        """
        Convert raw candidates list into structured findings.

        A nodule is flagged as significant if diameter >= SIGNIFICANT_SIZE_MM.
        All candidates are returned in `nodules`; flagged ones are also in
        `flagged_findings` for the report.
        """
        nodules = []
        flagged = []

        for c in candidates:
            diam = c["diameter_mm"]
            prob = c["probability"]
            loc  = (
                f"x={c['coordX_mm']} y={c['coordY_mm']} z={c['coordZ_mm']} mm"
            )

            # Density classification by size (proxy — no HU available here)
            if diam < 6:
                density = "subsolid/indeterminate (small)"
            elif diam < 20:
                density = "solid indeterminate"
            else:
                density = "solid — large"

            nodule_entry = {
                "size_mm":    diam,
                "location":   loc,
                "density":    density,
                "probability": prob,
                "logit":      c["logit"],
                "coordX_mm":  c["coordX_mm"],
                "coordY_mm":  c["coordY_mm"],
                "coordZ_mm":  c["coordZ_mm"],
            }
            nodules.append(nodule_entry)

            if diam >= SIGNIFICANT_SIZE_MM:
                flagged.append({
                    "finding":     f"Pulmonary nodule ≥ {SIGNIFICANT_SIZE_MM} mm",
                    "size_mm":     diam,
                    "location":    loc,
                    "probability": prob,
                    "threshold_mm": SIGNIFICANT_SIZE_MM,
                })

        return {
            "status":           "success",
            "error":            None,
            "nodules":          nodules,
            "flagged_findings": flagged,
            "logit_threshold":  self.logit_threshold,
        }