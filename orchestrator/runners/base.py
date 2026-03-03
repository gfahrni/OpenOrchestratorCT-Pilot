# base.py
# Abstract base class for all module runners.
# Every runner must implement run() and check_installation().

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """
    Abstract base class that defines the interface all runners must implement.

    Each runner is responsible for:
      - Verifying its external tool is installed and callable
      - Running that tool on a NIfTI input file
      - Returning a standardized findings dictionary
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The module-level config dict from config.yaml
                    (e.g., config["modules"]["totalsegmentator"])
        """
        self.config = config
        self.enabled = config.get("enabled", False)

    @abstractmethod
    def run(self, nifti_path: str, work_dir: str) -> dict:
        """
        Run the analysis tool on the given NIfTI file.

        Args:
            nifti_path: Absolute path to the input .nii.gz file
            work_dir:   Temporary working directory for intermediate outputs

        Returns:
            A dict with at minimum:
              {
                "status": "success" | "failed",
                "error":  str | None,
                ... module-specific findings ...
              }
        """
        pass

    @abstractmethod
    def check_installation(self) -> bool:
        """
        Verify that the external tool is installed and reachable.

        Returns:
            True if the tool is available, False otherwise.
        """
        pass