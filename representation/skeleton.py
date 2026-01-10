import numpy as np

class State:
    def __init__(self, joints, timestamp):
        self.timestamp = timestamp
        self.joints = joints  # Dict of joint: [x, y, z]
        self.joint_angles = self.compute_angles()  # Simplified angles
        self.joint_velocities = {}  # To be computed in trajectory
        self.center_of_mass = self.compute_com()

    def compute_angles(self):
        # Example: Compute angle between shoulder-elbow-wrist (simplified 2D projection)
        angles = {}
        for side in ['left', 'right']:
            shoulder = np.array(self.joints[f'{side}_shoulder'])
            elbow = np.array(self.joints[f'{side}_elbow'])
            wrist = np.array(self.joints[f'{side}_wrist'])
            vec1 = elbow - shoulder
            vec2 = wrist - elbow
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angles[f'{side}_elbow'] = angle
        return angles

    def compute_com(self):
        # Simple average of all joint positions
        positions = np.array(list(self.joints.values()))
        return np.mean(positions, axis=0).tolist()

class Trajectory:
    def __init__(self, states):
        self.states = states  # List of State objects
        self.compute_velocities()

    def compute_velocities(self):
        for i in range(1, len(self.states)):
            prev = self.states[i-1]
            curr = self.states[i]
            dt = curr.timestamp - prev.timestamp
            for joint in curr.joints:
                vel = (np.array(curr.joints[joint]) - np.array(prev.joints[joint])) / dt
                curr.joint_velocities[joint] = vel.tolist()

# Example usage (load from JSON and create Trajectory)
if __name__ == "__main__":
    import json
    with open('../storage/human_trajectory.json', 'r') as f:
        raw_data = json.load(f)
    states = [State(d['joints'], d['timestamp']) for d in raw_data]
    traj = Trajectory(states)
    print(traj.states[0].joint_angles)  # Test output