import numpy as np
from scipy.spatial.transform import Rotation as R

def pre_grasp_pose(grasp_pose, r, grasp_axis_local):
    # Extract position and orientation
    x, y, z, qx, qy, qz, qw = grasp_pose

    # Convert quaternion to rotation matrix
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()

    # Convert the grasp axis from local to world frame
    grasp_axis_world = rotation_matrix @ grasp_axis_local

    # Compute the offset position
    offset_position = np.array([x, y, z]) - r * grasp_axis_world

    # Create the pre-grasp pose
    pre_grasp_pose = [
        offset_position[0],
        offset_position[1],
        offset_position[2],
        qx, qy, qz, qw
    ]

    return pre_grasp_pose

# Example usage
grasp_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 1, 0]  # Example grasp pose
r = 1  # Offset distance
grasp_axis_local = np.array([0, 0, 1])  # Example grasp axis in local frame (z-axis in local frame)
pre_grasp_pose = pre_grasp_pose(grasp_pose, r, grasp_axis_local)
print("Pre-Grasp Pose:", pre_grasp_pose)