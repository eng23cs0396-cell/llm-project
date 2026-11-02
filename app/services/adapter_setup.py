from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)

ADAPTER_FILE_ID = "1vfhisRUh-LCl7HBj6uPAQ6bExlCqN6HU"
ADAPTER_URL = f"https://drive.google.com/uc?id={ADAPTER_FILE_ID}"


def _expected_adapter_paths(base_dir: Path) -> list[Path]:
    training_artifacts = base_dir / "training" / "artifacts"
    return [
        training_artifacts,
        training_artifacts / "flan_t5_sql_lora" / "final",
        training_artifacts / "mistral_7b_sql_lora" / "final",
    ]


def setup_adapters(base_dir: str | Path, *, force: bool = False) -> bool:
    """Download and extract adapter artifacts if they are missing.

    Returns True when artifacts are available at the end of the call.
    Returns False when setup could not complete.
    """
    root_dir = Path(base_dir).resolve()
    expected_paths = _expected_adapter_paths(root_dir)

    auto_download = os.getenv("AUTO_DOWNLOAD_ADAPTERS", "").lower() in ("1", "true", "yes")
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()

    if not force and all(path.exists() for path in expected_paths):
        logger.info("[ADAPTER_SETUP] Artifacts already present")
        return True

    if not force and not auto_download and nvidia_key:
        logger.info("[ADAPTER_SETUP] NVIDIA NIM configured; skipping local adapter zip download")
        return False

    zip_output = root_dir / "training" / "artifacts.zip"
    zip_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("[ADAPTER_SETUP] Downloading adapter artifacts from Google Drive")
        gdown.download(ADAPTER_URL, str(zip_output), quiet=False)

        logger.info("[ADAPTER_SETUP] Extracting %s", zip_output)
        with zipfile.ZipFile(zip_output, "r") as zip_ref:
            zip_ref.extractall(str(root_dir))

        # Support zip archives that unpack to ./artifacts instead of ./training/artifacts.
        root_artifacts = root_dir / "artifacts"
        training_artifacts = root_dir / "training" / "artifacts"
        if root_artifacts.exists() and not training_artifacts.exists():
            training_artifacts.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root_artifacts), str(training_artifacts))

        if all(path.exists() for path in expected_paths):
            logger.info("[ADAPTER_SETUP] Adapter setup complete")
            return True

        logger.warning("[ADAPTER_SETUP] Download/extract finished but expected adapter folders are still missing")
        return False
    except Exception as exc:
        logger.warning("[ADAPTER_SETUP] Failed to set up adapters: %s", exc)
        return False
