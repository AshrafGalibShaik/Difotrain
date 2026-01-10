import json
import random
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.skeleton import State, Trajectory
from storage.movement_db import MovementDB

def generate_dummy_trajectory():
    states = []
    # Joints needed
    JOINTS = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
              'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    for t in range(50):
        timestamp = t * 0.1
        joints = {}
        for joint in JOINTS:
            # Random position
            joints[joint] = [random.random(), random.random(), random.random()]
        
        state = State(joints, timestamp)
        states.append(state)
        
    return Trajectory(states)

if __name__ == "__main__":
    db = MovementDB()
    traj = generate_dummy_trajectory()
    print("Saving dummy trajectory...")
    db.save_trajectory(traj, "dummy_demo")
    print(f"DB now has {len(db.data)} entries.")
