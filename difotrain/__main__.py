#!/usr/bin/env python3
import sys
import argparse
import urllib.request
import os
import ssl
from pathlib import Path

# Adjust import to relative
try:
    from .capture.record_pose import main as record_main
except ImportError:
    # Fallback for running as script (though technically __main__ implies package execution)
    from capture.record_pose import main as record_main

def setup_mediapipe():
    print("Inspecting mp.tasks...")
    try:
        import mediapipe.tasks.python as tasks
        from mediapipe.tasks.python import vision
        print(f"Vision module: {vision}")
        print(f"Has PoseLandmarker: {hasattr(vision, 'PoseLandmarker')}")
    except Exception as e:
        print(f"Error inspecting tasks: {e}")
        import traceback
        traceback.print_exc()

    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    model_filename = "pose_landmarker_lite.task"
    
    # Download to current working directory
    model_path = Path.cwd() / model_filename

    if not model_path.exists():
        print(f"Downloading model from {model_url}...")
        try:
            # Create unverified context to avoid SSL errors on some systems
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(model_url, context=context) as response:
                with open(model_path, 'wb') as f:
                    f.write(response.read())
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"Failed to download model: {e}")
    else:
        print(f"Model already exists at {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Difo Human → Robot Motion Training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Record command
    record = subparsers.add_parser("record", help="Record human motion")
    
    # Setup command
    setup = subparsers.add_parser("setup", help="Download required MediaPipe models")

    args = parser.parse_args()

    if args.command == "record":
        record_main()
    elif args.command == "setup":
        setup_mediapipe()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()