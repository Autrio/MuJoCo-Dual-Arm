
import mujoco
import mujoco.viewer

import numpy as np
import trimesh
import argparse as ap
import sys
from icecream import ic
from scipy.spatial.transform import Rotation as R, Slerp
import scipy.spatial.transform as transform

from controllers.utils.QuinticPolynomial import create_quintic_trajectory, QuinticPolynomial


import cvxpy as cp
import copy


def Quat2rot(quat, informat: str, outformat: str, degrees: bool):
    if (informat == "xyzw"):
        r = R.from_quat(quat)
        return (r.as_euler(outformat, degrees))
    elif (informat == "wxyz"):
        tempQuat = quat[1:]
        tempQuat = np.append(tempQuat, quat[0])
        r = R.from_quat(tempQuat)
        return (r.as_euler(outformat, degrees))
    else:
        raise (ValueError)


def conv_parameters(model, data, jac_prev, dt=0.1, site='tip'):
    # jac_now = data.efc_J # current Jacobian
    jac_now = np.zeros((6, model.nv))
    force_site = model.site(site).id
    mujoco.mj_jacSite(model, data, jac_now[:3], jac_now[3:], force_site)

    C = data.qfrc_bias  # bias force: Coriolis, centrifugal, gravitational
    C = np.diag(C)
    jac_dot = (jac_now - jac_prev)/dt
    M_all = np.zeros((model.nv, model.nv))
    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_all, np.eye(model.nv))
    M_inv = M_all[:9, :9]
    Mx_inv = jac_now @ M_inv @ jac_now.T

    mm = np.zeros((model.nv, model.nv))  # mass matrix
    mujoco.mj_fullM(model, mm, data.qM)

    fext = np.zeros((6, ))
    # id = 0
    # mujoco.mj_contactForce(model, data, id, fext)
    # print("External Force : ", fext)
    # print("Joint Force : ", data.qfrc_applied.shape)

    return Mx_inv, mm, C, jac_now, jac_dot, fext


def convex_optimization(model, data, x_ref, xd_ref, xd, xdd_ref, fext, D, K, J, J_dot, Mx_inv, MM, C, dt):
    # * constants
    q_d = data.qvel
    x_pos = data.site('tip').xpos

    end_effector_quat = data.xquat[model.body('hand').id]
    # x_ori = Quat2rot(end_effector_quat, "wxyz", "xyz", True)
    x = np.concatenate((x_pos, end_effector_quat))
    error_x = x_ref - x
    error_xd = xd_ref - xd
    x_dd = xdd_ref + Mx_inv @ (D @ error_xd + K @ error_x - fext)

    # M = np.random.uniform(size=(7, 7)) # M = R^(7x7)
    # C = np.random.uniform(size=(7, 7)) # C = R^(7x7)

    # * variables
    q_dd = cp.Variable(9)  # q_dd = R^7

    # * bounds
    tau_max = np.random.uniform(size=(9))  # tau_max = R^7
    tau_min = np.random.uniform(size=(9))  # tau_min = R^7
    qdd_max = np.random.uniform(size=(9))  # qdd_max = R^7
    qdd_min = np.random.uniform(size=(9))  # qdd_min = R^7

    # min ||J@q_dd + J_d@q_d - x_dd||

    objective = cp.Minimize(cp.sum_squares(J @ q_dd + J_dot @ q_d - x_dd))
    constraints = [MM @ q_dd + C @ q_d <= tau_max,
                   MM @ q_dd + C @ q_d >= tau_min,
                   q_dd <= qdd_max,
                   q_dd >= qdd_min]

    problem = cp.Problem(objective, constraints)
    ic(problem.status)
    ic(problem.is_dcp())

    problem.solve()
    return q_dd.value


# * mujoco: wxyz, normal: xyzw
def mujoco2normal(q):
    return np.array([q[1], q[2], q[3], q[0]])

def normal2mujoco(q):
    return np.array([q[3], q[0], q[1], q[2]])


def slerp(x_init_quat, x_final_quat, final_n, n_steps):
    x_init_quat = mujoco2normal(x_init_quat)
    x_final_quat = mujoco2normal(x_final_quat)
    
    rot_times = np.array([0, final_n])
    rots = R.from_quat([x_init_quat, x_final_quat])
    
    slerp = Slerp(rot_times, rots)
    times = np.linspace(0, final_n, n_steps)
    quats = slerp(times).as_quat()
    quats = np.array([normal2mujoco(q) for q in quats])
    quat_dot = np.diff(quats, axis=0) / (final_n / n_steps)
    quat_ddot = np.diff(quat_dot, axis=0) / (final_n / n_steps)
    
    quat_dot_f, quat_ddot_f = np.zeros_like(quats), np.zeros_like(quats)
    quat_dot_f[:-1, :] = copy.deepcopy(quat_dot)
    quat_dot_f[-1:, :] = copy.deepcopy(quat_dot[-1, :])
    quat_ddot_f[:-2, :] = copy.deepcopy(quat_ddot)
    quat_ddot_f[-2:, :] = copy.deepcopy(quat_ddot[-2, :])
    
    return quats, quat_dot_f, quat_ddot_f 

def quintic_pos(init_pos, final_pos, final_n, nsteps):
    x_poly = QuinticPolynomial(init_pos[0], np.array([0.]), np.array([0.]),
                               final_pos[0], np.array([0.]), np.array([0.]), nsteps)
    y_poly = QuinticPolynomial(init_pos[1], np.array([0.]), np.array([0.]), final_pos[1],
                               np.array([0.]), np.array([0.]), nsteps)
    z_poly = QuinticPolynomial(init_pos[2], np.array([0.]), np.array([0.]),
                               final_pos[2], np.array([0.]), np.array([0.]), nsteps)

    times = np.linspace(0, final_n, nsteps)
    pos_ref = np.vstack([x_poly.calc_point(times),
                         y_poly.calc_point(times), 
                         z_poly.calc_point(times)]).T
    pos_dot_ref = np.vstack([x_poly.calc_first_derivative(times),
                             y_poly.calc_first_derivative(times),
                             z_poly.calc_first_derivative(times)]).T
    pos_ddot_ref = np.vstack([x_poly.calc_second_derivative(times),
                              y_poly.calc_second_derivative(times),
                              z_poly.calc_second_derivative(times)]).T

    return pos_ref, pos_dot_ref, pos_ddot_ref    
    
def main():
    model_path = '/home/faizal/Documents/MuJoCo-Dual-Arm/models/utils/franka_emika_panda/panda.xml'

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False)

    key_id = model.key("home").id

    dt = model.opt.timestep
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    force_site = model.site('tip').id

    end_effector_mat = data.site('tip').xmat
    end_effector_quat = np.zeros((4, ))
    mujoco.mju_mat2Quat(end_effector_quat, end_effector_mat)

    x_init = data.site(force_site).xpos.reshape(-1, 1)
    x_final = copy.deepcopy(x_init) + np.array([0.0, 0.0, -0.3]).reshape(-1, 1)
    
    quat_ref, quat_dot_ref, quat_ddot_ref = slerp(end_effector_quat, end_effector_quat, 500, 500)
    pos_ref, pos_dot_ref, pos_ddot_ref = quintic_pos(x_init, x_final, 500, 500)

    pose_ref = np.concatenate([pos_ref, quat_ref], axis=-1)
    pose_dot_ref = np.concatenate([pos_dot_ref, quat_dot_ref], axis=-1)
    pose_ddot_ref = np.concatenate([pos_ddot_ref, quat_ddot_ref], axis=-1)

    ic(pose_ref.shape, pose_dot_ref.shape, pose_ddot_ref.shape)

    jac_prev = np.zeros((6, model.nv))
    counter = 0

    K = np.diag([100, 100, 100, 100, 100, 100])
    D = 2 * np.sqrt(K)

    while viewer.is_running():
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        Mx_inv, MM, C, J, J_dot, fext = conv_parameters(model, data, jac_prev)

        jac_prev = copy.deepcopy(J)

        xd = jac_prev @ data.qvel
        
        q_dd = convex_optimization(model, 
                            data, 
                            pose_ref[counter],
                            pose_dot_ref[counter],
                            xd,
                            pose_ddot_ref[counter],
                            fext,
                            D, K, J, J_dot, Mx_inv, MM, C, dt)
        
        viewer.sync()
        print("=" * 50)
        counter += 1
    
    


if __name__ == "__main__":
    main()
