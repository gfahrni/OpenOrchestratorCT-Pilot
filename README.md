# Thoracic CT Orchestrator

A Python orchestrator that integrates three open-source AI tools to analyze thoracic CT scans (NIfTI format) and generate a structured findings report.

---

## What it does

Takes a `.nii.gz` file as input and runs up to three analysis modules (configurable):

1. **TotalSegmentator** — segments anatomical structures and computes volumes for cardiomegaly (heart), goitre (thyroid), and adrenal findings
2. **Emphysema Quantification** — quantifies emphysema via low-attenuation area (LAA) percentage
3. **Lung Nodule Detection** — detects and characterizes pulmonary nodules

Outputs a `.txt` report summarizing all findings.

---

## Project Structure

```
orchestrator-test/
├── run.py                        # Main entrypoint
├── config.yaml                   # Enable/disable modules + paths + conda env names
├── requirements.txt              # Orchestrator dependencies only
├── README.md
├── .gitignore                    # Excludes modules/, data-ct-nifti/, and outputs/
│
├── envs/                         # Conda environment files (one per module)
│   ├── totalseg_env.yaml
│   ├── emphysema_env.yaml
│   └── nodules_env.yaml
│
├── orchestrator/
│   ├── __init__.py
│   ├── pipeline.py               # Coordinates runners
│   ├── report.py                 # Assembles final .txt report
│   └── runners/
│       ├── base.py               # Abstract base runner
│       ├── totalsegmentator.py
│       ├── emphysema.py
│       └── nodule.py
│
├── modules/                      # NOT tracked by git — clone the 3 repos here
│   ├── TotalSegmentator/
│   ├── Emphysema_Quantification/
│   └── LungNoduleDetection/
│
├── data-ct-nifti/                # NOT tracked by git — place your .nii.gz files here
│   └── CT_PE_017.nii.gz
│
└── outputs/                      # NOT tracked by git — generated reports land here
```

---

## Requirements

- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) — required to manage separate CUDA/PyTorch versions per module
- NVIDIA GPU (for HPC/workstation) or Apple Silicon Mac (MPS)

---

## Installation

### Step 1 — Clone this repo

```bash
git clone https://github.com/yourname/orchestrator-test.git
cd orchestrator-test
```

### Step 2 — Create the orchestrator conda environment

This is the environment you will always have active when running the orchestrator.
It only contains the lightweight dependencies needed to coordinate the pipeline.

```bash
conda create -n orchestrator python=3.11
conda activate orchestrator
pip install -r requirements.txt
```

> **VS Code users**: after creating the env, select it as your Python interpreter.
> Click the interpreter selector in the bottom-right corner of VS Code and choose
> the one pointing to `.../envs/orchestrator/bin/python`.
> VS Code will then automatically activate it in every new integrated terminal.

### Step 3 — Clone the three analysis modules into `modules/`

```bash
mkdir modules && cd modules
git clone https://github.com/wasserth/TotalSegmentator
git clone https://github.com/bdrad/Emphysema_Quantification
git clone https://github.com/rlsn/LungNoduleDetection
cd ..
```

### Step 4 — Create a conda environment for each module

Each module requires its own isolated environment to avoid CUDA/PyTorch version conflicts.
The orchestrator calls them internally via `conda run` — you never need to activate them manually.

```bash
conda env create -f envs/totalseg_env.yaml
conda env create -f envs/emphysema_env.yaml
conda env create -f envs/nodules_env.yaml
```

> Follow each module's own README for any additional setup steps (e.g. model weight downloads).

---

## Configuration

Edit `config.yaml` to enable/disable modules and configure device and paths.
The `device` option controls which hardware is used for inference:

```yaml
modules:
  totalsegmentator:
    enabled: true
    conda_env: "totalseg"
    # device options:
    #   "cpu"   — no GPU, slow but always works
    #   "gpu"   — NVIDIA GPU (CUDA)
    #   "gpu:1" — specific NVIDIA GPU by index
    #   "mps"   — Apple Silicon (M1/M2/M3)
    device: "mps"

  emphysema:
    enabled: false
    conda_env: "emphysema"
    script_path: "modules/Emphysema_Quantification/run.py"
    device: "cpu"

  lung_nodule:
    enabled: false
    conda_env: "nodules"
    script_path: "modules/LungNoduleDetection/detect.py"
    checkpoint: "modules/LungNoduleDetection/weights/model.pth"
    device: "cpu"

output:
  dir: "./outputs"
```

---

## Usage

Always run from the project root with the orchestrator environment active:

```bash
conda activate orchestrator
python run.py --input data-ct-nifti/CT_PE_017.nii.gz
```

The report will be saved to `outputs/CT_PE_017_report.txt`.


## Modules & Licenses

| Module | License | Source |
|--------|---------|--------|
| TotalSegmentator | Apache 2.0 | [GitHub](https://github.com/wasserth/TotalSegmentator) |
| Emphysema Quantification | MIT | [GitHub](https://github.com/bdrad/Emphysema_Quantification) |
| Lung Nodule Detection | MIT | [GitHub](https://github.com/rlsn/LungNoduleDetection) |

This orchestrator is distributed under the MIT License.

---

## Notes

- `modules/`, `data-ct-nifti/`, and `outputs/` are excluded from git via `.gitignore`
- Normative volume thresholds for TotalSegmentator findings are based on:
  Remark et al., Radiology AI 2025 — https://pubs.rsna.org/doi/10.1148/ryai.250506