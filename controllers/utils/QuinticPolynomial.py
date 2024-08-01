import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def pre_grasp_pose(grasp_pose, r, grasp_axis_local):
    # Extract position and orientation
    ([x, y, z], [qx, qy, qz, qw]) = grasp_pose

    # Convert quaternion to rotation matrix
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()

    # Convert the grasp axis from local to world frame
    grasp_axis_world = rotation_matrix @ grasp_axis_local

    # Compute the offset position
    offset_position = np.array([x, y, z]) - r * grasp_axis_world

    # Create the pre-grasp pose
    pre_grasp_pose = ([
        offset_position[0],
        offset_position[1],
        offset_position[2]],
        [qx, qy, qz, qw])

    return pre_grasp_pose

class QuinticPolynomial:
    def __init__(self, init_pos, init_vel, init_accel, final_pos, final_vel, final_accel, dist):
        # Derived coefficients
        self.a_0 = init_pos
        self.a_1 = init_vel
        self.a_2 = init_accel / 2.0

        # Solve the linear equation (Ax = B)
        A = np.array([[dist ** 3,      dist ** 4,        dist ** 5], 
                     [3 * dist ** 2,   4 * dist ** 3,    5 * dist ** 4],
                     [6 * dist,        12 * dist ** 2,   20 * dist ** 3]])

        B = np.array([final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2, 
                      final_vel - self.a_1 - 2 * self.a_2 * dist,
                      final_accel - 2 * self.a_2])

        x = np.linalg.solve(A, B)

        self.a_3 = x[0]
        self.a_4 = x[1]
        self.a_5 = x[2]

    def calc_point(self, s):
        xs = self.a_0 + self.a_1 * s + self.a_2 * s ** 2 + self.a_3 * s ** 3 + self.a_4 * s ** 4 + self.a_5 * s ** 5
        return xs

    def calc_first_derivative(self, s):
        xs = self.a_1 + (2 * self.a_2 * s) + (3 * self.a_3 * (s ** 2)) + \
        (4 * self.a_4 * (s ** 3)) + (5 * self.a_5 * (s ** 4))
        return xs

    def calc_second_derivative(self, s):
        xs = 2 * self.a_2 + 6 * self.a_3 * s + 12 * self.a_4 * (s ** 2) + 20 * self.a_5 * (s ** 3)
        return xs

    def calc_third_derivative(self, s):
        xs = 6 * self.a_3 + 24 * self.a_4 * s + 60 * self.a_5 * (s ** 2)
        return xs

def create_quintic_trajectory(init_pose, final_pose, steps):
    traj = []

    # Position interpolation
    x_poly = QuinticPolynomial(init_pose[0][0], 0, 0, final_pose[0][0], 0, 0, steps)
    y_poly = QuinticPolynomial(init_pose[0][1], 0, 0, final_pose[0][1], 0, 0, steps)
    z_poly = QuinticPolynomial(init_pose[0][2], 0, 0, final_pose[0][2], 0, 0, steps)

    # Quaternion interpolation
    qx_poly = QuinticPolynomial(init_pose[1][0], 0, 0, final_pose[1][0], 0, 0, steps)
    qy_poly = QuinticPolynomial(init_pose[1][1], 0, 0, final_pose[1][1], 0, 0, steps)
    qz_poly = QuinticPolynomial(init_pose[1][2], 0, 0, final_pose[1][2], 0, 0, steps)
    qw_poly = QuinticPolynomial(init_pose[1][3], 0, 0, final_pose[1][3], 0, 0, steps)

    for i in range(steps):
        pos = [x_poly.calc_point(i), y_poly.calc_point(i), z_poly.calc_point(i)]
        quat = [qx_poly.calc_point(i), qy_poly.calc_point(i), qz_poly.calc_point(i), qw_poly.calc_point(i)]
        quat /= np.linalg.norm(quat)  # Normalize quaternion to ensure it remains valid
        traj.append(pos + quat.tolist())

    
    return traj, 

def generate_end_effector_trajectories(object_trajectory, current_obj_pose, current_pose_L, current_pose_R):
    AtrajL = []
    AtrajR = []

    # Extract current object pose
    obj_pos = np.array(current_obj_pose[0])
    obj_quat = np.array(current_obj_pose[1])

    # Extract current end effector poses
    pos_L = np.array(current_pose_L[0])
    quat_L = np.array(current_pose_L[1])
    pos_R = np.array(current_pose_R[0])
    quat_R = np.array(current_pose_R[1])

    # Calculate inverse object rotation matrix
    inv_obj_R = quaternion_to_rotation_matrix(quat_inverse(obj_quat))

    # Calculate transformations from the object to the end effectors
    obj_to_ee_L_pos = inv_obj_R @ (pos_L - obj_pos)
    obj_to_ee_L_quat = quaternion_multiply(quat_inverse(obj_quat), quat_L)

    obj_to_ee_R_pos = inv_obj_R @ (pos_R - obj_pos)
    obj_to_ee_R_quat = quaternion_multiply(quat_inverse(obj_quat), quat_R)

    obj_to_ee_L = np.concatenate([obj_to_ee_L_pos, obj_to_ee_L_quat])
    obj_to_ee_R = np.concatenate([obj_to_ee_R_pos, obj_to_ee_R_quat])

    for obj_pose in object_trajectory:
        pos = np.array(obj_pose[:3])
        quat = np.array(obj_pose[3:])
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(quat)

        # Left end effector pose
        ee_L_pos = pos + R @ obj_to_ee_L[:3]
        ee_L_quat = quaternion_multiply(quat, obj_to_ee_L[3:])
        AtrajL.append(ee_L_pos.tolist() + ee_L_quat.tolist())

        # Right end effector pose
        ee_R_pos = pos + R @ obj_to_ee_R[:3]
        ee_R_quat = quaternion_multiply(quat, obj_to_ee_R[3:])
        AtrajR.append(ee_R_pos.tolist() + ee_R_quat.tolist())

    return AtrajL, AtrajR


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def quat_inverse(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])