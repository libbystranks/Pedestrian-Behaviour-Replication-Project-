# === visualise_smpl_motion.py ===
import torch
import smplx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

SMPL_MODEL_PATH = "/Users/libbystranks/Desktop/GitHub/ml-comotion/src/comotion_demo/data"
model = smplx.create(model_path=SMPL_MODEL_PATH, model_type='smpl', gender='NEUTRAL')

data = torch.load("ptFiles/pipeline_output.pt", map_location='cpu')
pose_data = data['pose'].float()
trans = data['trans'].float()
betas = data['betas'].float()

output = model(body_pose=pose_data[:, 3:], global_orient=pose_data[:, :3], transl=trans, betas=betas)
joint_seq = output.joints[:, :24].detach().cpu().numpy()

# SMPL 24-joint skeleton bone connections (parent → child)
BONES = [
    (0, 1), (0, 2), (0, 3),
    (3, 6), (6, 9), (9, 12),
    (12, 15),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]

all_joints = np.vstack(joint_seq)
X_MIN, X_MAX = all_joints[:, 0].min(), all_joints[:, 0].max()
Y_MIN, Y_MAX = all_joints[:, 1].min(), all_joints[:, 1].max()
Z_MIN, Z_MAX = all_joints[:, 2].min(), all_joints[:, 2].max()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_frame(i):
    ax.cla()
    joints = joint_seq[i]
    for a, b in BONES:
        ax.plot(
            [joints[a][0], joints[b][0]],
            [joints[a][1], joints[b][1]],
            [joints[a][2], joints[b][2]],
            c='blue'
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red')
    ax.set_xlim(X_MIN - 0.5, X_MAX + 0.5)
    ax.set_ylim(Y_MIN - 0.5, Y_MAX + 0.5)
    ax.set_zlim(Z_MIN - 0.5, Z_MAX + 0.5)
    ax.set_title(f"Frame {i}")

ani = animation.FuncAnimation(fig, plot_frame, frames=len(joint_seq), interval=100)
plt.show()
