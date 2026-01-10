import cv2
import mediapipe as mp
import json
import time
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Joints to track (subset for simplicity; MediaPipe provides 33 landmarks)
JOINTS = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

def get_joint_positions(landmarks):
    joints = {}
    for joint in JOINTS:
        idx = mp_pose.PoseLandmark[joint.upper()].value
        lm = landmarks.landmark[idx]
        joints[joint] = [lm.x, lm.y, lm.z]  # Normalized [x, y, z]; scale to real-world if needed
    return joints

# Capture loop
cap = cv2.VideoCapture(0)  # Webcam
trajectory = []  # List of states
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract and store state
            timestamp = time.time() - start_time
            joints = get_joint_positions(results.pose_landmarks)
            state = {
                "timestamp": timestamp,
                "joints": joints
            }
            trajectory.append(state)

        # Display
        cv2.imshow('Pose Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# Save trajectory to JSON
with open('../storage/human_trajectory.json', 'w') as f:
    json.dump(trajectory, f, indent=4)

print(f"Captured {len(trajectory)} states.")