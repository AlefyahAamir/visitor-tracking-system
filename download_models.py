"""Download the two model files required by VisitorTrackingSystem."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


MODELS = (
    {
        "name": "YOLO11s",
        "path": MODEL_DIR / "yolo11s.pt",
        "sha256": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
        "minimum_size": 18_000_000,
        "urls": (
            "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt?download=true",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        ),
    },
    {
        "name": "OSNet x0.25 MSMT17",
        "path": MODEL_DIR / "osnet_x0_25_msmt17.pth",
        "sha256": "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18",
        "minimum_size": 8_000_000,
        "urls": (
            "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
            "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_"
            "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
            "?download=true",
        ),
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_model(path: Path, minimum_size: int, expected_sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size < minimum_size:
        return False
    return file_sha256(path).lower() == expected_sha256.lower()


def _progress(block_number: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        downloaded_mb = block_number * block_size / (1024 * 1024)
        print(f"\rDownloaded {downloaded_mb:.1f} MB", end="", flush=True)
        return

    downloaded = min(block_number * block_size, total_size)
    percentage = downloaded * 100 / total_size
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(
        f"\r{percentage:6.1f}%  {downloaded_mb:.1f}/{total_mb:.1f} MB",
        end="",
        flush=True,
    )


def download_from_urls(urls: Iterable[str], destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    temporary_path.unlink(missing_ok=True)

    last_error: Exception | None = None

    for url in urls:
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "VisitorTrackingSystem/1.0"
                    )
                },
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                total_size = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0

                with temporary_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percentage = min(downloaded * 100 / total_size, 100.0)
                            print(
                                f"\r{percentage:6.1f}%  "
                                f"{downloaded / (1024 * 1024):.1f}/"
                                f"{total_size / (1024 * 1024):.1f} MB",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rDownloaded {downloaded / (1024 * 1024):.1f} MB",
                                end="",
                                flush=True,
                            )

            print()
            os.replace(temporary_path, destination)
            return

        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            print(f"\nDownload source failed: {exc}")

    raise RuntimeError(
        f"Could not download {destination.name}. Last error: {last_error}"
    )


def ensure_model_files() -> None:
    """Download missing or invalid model files and verify their checksums."""

    for model in MODELS:
        path: Path = model["path"]
        name: str = model["name"]
        expected_sha256: str = model["sha256"]
        minimum_size: int = model["minimum_size"]

        if is_valid_model(path, minimum_size, expected_sha256):
            print(f"Found valid {name}: models\\{path.name}")
            continue

        if path.exists():
            print(f"Replacing incomplete or invalid file: models\\{path.name}")
            path.unlink(missing_ok=True)

        print(f"Downloading {name}...")
        download_from_urls(model["urls"], path)

        if not is_valid_model(path, minimum_size, expected_sha256):
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded {path.name}, but its validation failed. "
                "Run download_models.py again."
            )

        print(f"Saved models\\{path.name}")


if __name__ == "__main__":
    try:
        ensure_model_files()
        print("All required model files are ready.")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
