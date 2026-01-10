import json
import os

class MovementDB:
    def __init__(self, db_path='movements.json'):
        self.db_path = db_path
        self.data = self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return []

    def save_trajectory(self, trajectory, label):
        # Serialize Trajectory to dict
        traj_dict = {
            "label": label,
            "states": [
                {
                    "timestamp": s.timestamp,
                    "joints": s.joints,
                    "joint_angles": s.joint_angles,
                    "joint_velocities": s.joint_velocities,
                    "center_of_mass": s.center_of_mass
                } for s in trajectory.states
            ]
        }
        self.data.append(traj_dict)
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=4)

# Example
if __name__ == "__main__":
    # Assume traj from skeleton.py
    db = MovementDB()
    db.save_trajectory(traj, "walking_demo")