# motion_loader.py
import os
import torch
import numpy as np

def load_pose_data_from_folder(folder_path):
    """
    Load all .pt files in the folder and extract pose vectors from them.

    Returns:
        pose_list: List of (filename, frame_index, pose_vector) tuples
        pose_matrix: Numpy array of shape (total_frames, D)
    """
    pose_list = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pt"):
            file_path = os.path.join(folder_path, filename)
            data = torch.load(file_path)

            # Assuming data is a dict with key 'poses' containing shape (num_frames, D)
            poses = data['pose']  # Adjust this if your structure is different

            for i, pose in enumerate(poses):
                pose_np = pose.detach().cpu().numpy().astype(np.float32)
                pose_list.append((filename, i, pose_np))

    pose_matrix = np.stack([p[2] for p in pose_list], axis=0)  # (N, D)

    return pose_list, pose_matrix
if __name__ == "__main__":
    data_folder = "ptFiles"
    poses, id_map = load_pose_data_from_folder(data_folder)
    print(f"Loaded {len(poses)} poses from {len(id_map)} files.")
    print(f"Shape of first pose: {poses[0][2].shape}")



