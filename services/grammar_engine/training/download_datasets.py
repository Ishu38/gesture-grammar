"""MLAF Training Pipeline — Dataset Download.

Downloads publicly available hand-gesture landmark datasets:
  1. Zenodo Hand Landmarks CSV (CC-BY-4.0, ~1 MB)
  2. HaGRID annotation JSONs (pre-extracted MediaPipe landmarks, ~200 MB subset)

Usage:
    python -m training.download_datasets
    python training/download_datasets.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import (
    HAGRID_ANNOTATIONS_ZIP_URL,
    HAGRID_GESTURE_CLASSES,
    RAW_DIR,
    ZENODO_HAND_LANDMARKS_URL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_USER_AGENT = "MLAF-Training-Pipeline/1.0 (research; +https://github.com/mlaf)"
_CHUNK_SIZE = 64 * 1024  # 64 KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, *, description: str = "") -> bool:
    """Download *url* to *dest* with progress logging. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("  ✓ Already exists: %s (%d bytes)", dest.name, dest.stat().st_size)
        return True

    label = description or dest.name
    logger.info("  Downloading %s …", label)
    logger.info("    URL: %s", url)

    req = Request(url, headers={"User-Agent": _USER_AGENT})
    t0 = time.perf_counter()

    try:
        with urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)

            elapsed = time.perf_counter() - t0
            size_mb = downloaded / (1024 * 1024)
            logger.info(
                "    ✓ Saved %s (%.2f MB in %.1fs)",
                dest.name, size_mb, elapsed,
            )
            return True

    except (HTTPError, URLError, OSError) as exc:
        logger.error("    ✗ Failed to download %s: %s", label, exc)
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Source 1: Zenodo Hand Landmarks
# ---------------------------------------------------------------------------

def download_zenodo_landmarks() -> Path | None:
    """Download the Zenodo hand-gesture landmarks CSV.

    Format: CSV with 63 landmark columns (21 × x,y,z) + label.
    License: CC-BY-4.0.
    """
    logger.info("=== Source 1: Zenodo Hand Landmarks ===")
    dest = RAW_DIR / "zenodo_hand_landmarks.csv"

    ok = _download_file(
        ZENODO_HAND_LANDMARKS_URL,
        dest,
        description="Zenodo hand-gestures.csv",
    )
    return dest if ok else None


# ---------------------------------------------------------------------------
# Source 2: HaGRID Annotations
# ---------------------------------------------------------------------------

def download_hagrid_annotations() -> list[Path]:
    """Download HaGRID annotation ZIP from Sbercloud OBS and extract.

    The ZIP contains per-gesture JSON files with bounding boxes and labels
    for 34 gesture classes (we use a subset).
    """
    logger.info("=== Source 2: HaGRID Annotations (Sbercloud OBS) ===")
    hagrid_dir = RAW_DIR / "hagrid"
    hagrid_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_jsons = sorted(hagrid_dir.glob("*.json"))
    if len(existing_jsons) >= 5:
        logger.info("  HaGRID annotations already extracted (%d JSON files)", len(existing_jsons))
        return existing_jsons

    # Download ZIP
    zip_path = RAW_DIR / "hagrid_annotations.zip"
    ok = _download_file(
        HAGRID_ANNOTATIONS_ZIP_URL,
        zip_path,
        description="HaGRID annotations.zip (~200 MB)",
    )

    if not ok:
        logger.error("  Failed to download HaGRID annotations ZIP")
        return []

    # Extract
    logger.info("  Extracting HaGRID annotations …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List contents to understand structure
            names = zf.namelist()
            json_names = [n for n in names if n.endswith(".json")]
            logger.info("    ZIP contains %d files (%d JSON)", len(names), len(json_names))

            for name in json_names:
                # Extract to hagrid_dir, flattening any subdirectory structure
                basename = Path(name).name
                dest = hagrid_dir / basename
                if not dest.exists():
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

        extracted = sorted(hagrid_dir.glob("*.json"))
        logger.info("    Extracted %d JSON files to %s", len(extracted), hagrid_dir)

        # Clean up ZIP to save disk space
        zip_path.unlink()
        logger.info("    Deleted ZIP (saved disk space)")

        return extracted

    except (zipfile.BadZipFile, OSError) as exc:
        logger.error("    Failed to extract ZIP: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _report_dataset_stats(zenodo_path: Path | None, hagrid_paths: list[Path]) -> dict:
    """Generate a summary dict for logging."""
    stats: dict = {"zenodo": None, "hagrid": []}

    if zenodo_path and zenodo_path.exists():
        # Count rows (excluding header)
        with open(zenodo_path, "r") as f:
            n_lines = sum(1 for _ in f) - 1
        stats["zenodo"] = {
            "file": str(zenodo_path),
            "size_bytes": zenodo_path.stat().st_size,
            "num_samples": max(0, n_lines),
        }
        logger.info("Zenodo: %d samples", stats["zenodo"]["num_samples"])

    for hp in hagrid_paths:
        if hp.exists():
            try:
                with open(hp) as f:
                    data = json.load(f)
                n = len(data) if isinstance(data, (list, dict)) else 0
            except (json.JSONDecodeError, OSError):
                n = 0
            stats["hagrid"].append({
                "file": str(hp),
                "size_bytes": hp.stat().st_size,
                "num_entries": n,
            })

    total_hagrid = sum(h["num_entries"] for h in stats["hagrid"])
    logger.info("HaGRID: %d total annotation entries across %d files", total_hagrid, len(stats["hagrid"]))
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Download all datasets. Returns summary stats dict."""
    logger.info("MLAF Training Pipeline — Dataset Download")
    logger.info("Target directory: %s", RAW_DIR)

    zenodo_path = download_zenodo_landmarks()
    hagrid_paths = download_hagrid_annotations()

    stats = _report_dataset_stats(zenodo_path, hagrid_paths)
    logger.info("Download complete.")
    return stats


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)
