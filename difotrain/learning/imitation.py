import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from representation.skeleton import Trajectory, State  # Import your Trajectory

class MotionDataset(Dataset):
    def __init__(self, trajectories):
        self.inputs = []  # Human states (e.g., joint positions)
        self.targets = []  # "Robot" actions (simulated as shifted human angles for demo)
        for traj in trajectories:
            for state in traj.states:
                # Flatten joints: values is list of [x,y,z], so we make it array and flatten
                input_vec = np.array(list(state.joints.values())).flatten()
                target_vec = np.array(list(state.joint_angles.values())).flatten()
                self.inputs.append(input_vec)
                self.targets.append(target_vec)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

def train_imitation(trajectories, epochs=100):
    dataset = MotionDataset(trajectories)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = dataset.inputs.shape[1]  # e.g., num_joints * 3
    output_size = dataset.targets.shape[1]  # num_angles
    model = ImitationModel(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'imitation_model.pth')
    return model

# Example (load from DB and train)
if __name__ == "__main__":
    from storage.movement_db import MovementDB
    db = MovementDB()
    trajectories = []  # Load and convert db.data to Trajectory objects
    for entry in db.data:
        states = [State(s['joints'], s['timestamp']) for s in entry['states']]  # Reconstruct
        trajectories.append(Trajectory(states))
    model = train_imitation(trajectories)