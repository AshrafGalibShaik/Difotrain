#!/usr/bin/env python3
"""Package entry point. Delegates to :mod:`difotrain.cli`; keeps the MediaPipe
model downloader here so ``difotrain setup`` has no heavy imports.
"""
import ssl
import sys
import urllib.request
from pathlib import Path


def setup_mediapipe() -> None:
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    model_filename = "pose_landmarker_lite.task"
    model_path = Path.cwd() / model_filename

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return

    print(f"Downloading model from {model_url}...")
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(model_url, context=context) as response:
            model_path.write_bytes(response.read())
        print(f"Model downloaded successfully to {model_path}")
    except Exception as e:  # pragma: no cover - network
        print(f"Failed to download model: {e}")


def main() -> None:
    from .cli import main as cli_main

    sys.exit(cli_main())


if __name__ == "__main__":
    main()
