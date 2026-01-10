import cv2
import mediapipe as mp
import json
import time
import numpy as np
from pathlib import Path

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

def main():
    print("Initializing MediaPipe Pose Landmarker...")
    
    # Model path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / 'pose_landmarker_lite.task'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run setup_mediapipe.py first.")
        return

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
        return

    # mp_drawing = mp.solutions.drawing_utils # Helper for drawing (might still exist in basic mp)
    # Note: drawing_utils might expect protobuf landmarks, tasks api returns object. 
    # For now, we'll skip complex drawing or implement simple drawing to avoid incompatibilities.
    # Actually, let's keep it simple and just capture for now.

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

    # Save
    storage_dir = project_root / 'storage'
    storage_dir.mkdir(exist_ok=True)
    output_path = storage_dir / 'human_trajectory.json'
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=4)

    print(f"Captured {len(trajectory)} states.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()