import cv2
import mediapipe as mp
import json
import time
import numpy as np
from pathlib import Path
import sys

# New API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_joint_positions(pose_landmarks):
    # Joints to track (subset for simplicity)
    # Task API landmarks have same indices as legacy for Pose (BlazePose)
    # 0 - nose, 11 - left_shoulder, 12 - right_shoulder, ...
    JOINTS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    joints = {}
    for name, idx in JOINTS.items():
        if idx < len(pose_landmarks):
            lm = pose_landmarks[idx]
            joints[name] = [lm.x, lm.y, lm.z]
    return joints

def capture_trajectory():
    """Capture a pose trajectory from the webcam and return it as a list of
    ``{"timestamp", "joints"}`` frames. Shared by :func:`main` (which also saves
    to disk) and by the human-video data source.
    """
    return _run_capture()


def main():
    """CLI entry point: capture and save to ``storage/human_trajectory.json``."""
    trajectory = _run_capture()
    storage_dir = Path.cwd() / 'storage'
    storage_dir.mkdir(exist_ok=True)
    output_path = storage_dir / 'human_trajectory.json'
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=4)
    print(f"Captured {len(trajectory)} states.")
    print(f"Saved to {output_path}")
    return trajectory


def _run_capture():
    print("Initializing MediaPipe Pose Landmarker...")

    # 1. Check Current Working Directory first
    model_filename = 'pose_landmarker_lite.task'
    cwd_model = Path.cwd() / model_filename
    
    if cwd_model.exists():
        model_path = cwd_model
    else:
        # 2. Check package directory (fallback)
        package_root = Path(__file__).resolve().parent.parent
        pkg_model = package_root / model_filename
        if pkg_model.exists():
             model_path = pkg_model
        else:
            print(f"Error: Model not found at {cwd_model} or {pkg_model}")
            print("Please run 'difotrain setup' first to download the model.")
            return []

    print(f"Using model: {model_path}")

    # Create options
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO)
    
    # Create landmarker
    try:
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Failed to create PoseLandmarker: {e}")
        return []

    # Capture loop
    cap = cv2.VideoCapture(0)
    trajectory = []
    start_time = time.time()
    
    print("Starting capture. Press 'q' to stop.")

    try:
        if not cap.isOpened():
            print("Error: Could not open video capture device.")
            landmarker.close()
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Timestamp in ms
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            # Detect
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.pose_landmarks:
                # Assuming single person
                landmarks = detection_result.pose_landmarks[0]
                
                # Visual feedback (simple circles)
                for lm in landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Store state
                joints = get_joint_positions(landmarks)
                state = {
                    "timestamp": timestamp_ms / 1000.0,
                    "joints": joints
                }
                trajectory.append(state)

            # Display
            cv2.imshow('Pose Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Capture stopped by user.")
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    return trajectory


if __name__ == "__main__":
    main()