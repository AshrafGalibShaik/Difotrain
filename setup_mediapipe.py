import mediapipe as mp
import urllib.request
import os
import ssl

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
model_path = "pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print(f"Downloading model from {model_url}...")
    try:
        # Create unverified context to avoid SSL errors on some systems
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(model_url, context=context) as response:
            with open(model_path, 'wb') as f:
                f.write(response.read())
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")
else:
    print("Model already exists.")
