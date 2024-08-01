
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
    x_ori = Quat2rot(end_effector_quat, "wxyz", "xyz", True)
    x = np.concatenate((x_pos, x_ori))
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

    ic(q_dd.value)

    return q_dd.value


model_path = '/home/faizal/Documents/MuJoCo-Dual-Arm/models/utils/franka_emika_panda/panda.xml'

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False)

key_id = model.key("home").id

mujoco.mj_resetDataKeyframe(model, data, key_id)

mm = np.zeros((model.nv, model.nv))
dt = model.opt.timestep
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)
force_site = model.site('tip').id

# ic(data.site(force_site).xpos, data.site(force_site).xquat)
# ic(data.xquat.shape)
eef_body_index = model.body('hand').id
end_effector_mat = data.site('tip').xmat
end_effector_quat = np.zeros((4, ))
mujoco.mju_mat2Quat(end_effector_quat, end_effector_mat)
ic(end_effector_quat)

# x_ori = Quat2rot(end_effector_quat, "wxyz", "xyz", True)
# ic(x_ori)

# exit()
x_init = data.site(force_site).xpos.reshape(-1, 1)
x_final = copy.deepcopy(x_init) + np.array([0.0, 0.0, -0.3]).reshape(-1, 1)
# # ic(x_init, x_final)
# t = np.linspace(0, 1, 10)

# xpos_t = x_init * (1-t)  + x_final * t
# xpos_t = xpos_t.T

# xori_t = np.zeros((10, 3))
# x_t = np.concatenate((xpos_t, xori_t), axis=1)
# ic(x_t.shape)
# # exit()

# xd_t = np.zeros((100, 6))
# xdd_t = np.zeros((100, 6))

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
    return quats


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

# ic(slerp([1,0,0,0], [0,1,0,0], 100).shape)


ic(end_effector_quat.shape)

quat_trajs = slerp(end_effector_quat, end_effector_quat, 500, 500)
pos_ref, pos_dot_ref, pos_ddot_ref = quintic_pos(x_init, x_final, 500, 500)
ic(quat_trajs.shape)
ic(pos_ref.shape, pos_dot_ref.shape, pos_ddot_ref.shape)
exit()

x_init_pos = data.site(force_site).xpos
x_init_quat = end_effector_quat
x_init = (list(x_init_pos), list(x_init_quat))

x_final_pos = copy.deepcopy(x_init_pos) + np.array([0.0, 0.0, -0.3])
x_final_quat = x_init_quat
x_final = (list(x_final_pos), list(x_final_quat))

x_traj = create_quintic_trajectory(x_init, x_final, 10)
x_traj = np.array(x_traj)
ic(x_traj)

x_traj_pos = x_traj[:, :3]
x_traj_quat = slerp(x_init_quat, x_final_quat, 10)

# exit()

# Compute the full mass matrix from the compressed qM

jac_prev = np.zeros((6, model.nv))
counter = 0

while viewer.is_running():
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

    Mx_inv, MM, C, J, J_dot, fext = conv_parameters(model, data, jac_prev)

    jac_prev = copy.deepcopy(J)

    xd = jac_prev @ data.qvel

    # ic(M.shape, C.shape, J.shape, J_dot.shape, fext.shape)

    # print(f"nv: {model.nv}")
    # print(data.qM.shape)
    # mujoco.mj_fullM(model, mm, data.qM)
    # print(f"mm: {mm.shape} det: {np.linalg.det(mm)}")
    # # print(f"data: {data.contact}")

    # print(data.contact)
    # for id,c in enumerate(data.contact):
    #     print("contact:", id, c)

    # M, C, J, J_dot, fext = conv_parameters(data.efc_J)

    # convex_optimization( x_ref, xd_ref, xd, xdd_ref, fext, D, K, J, M, C, dt)

    K = np.diag([100, 100, 100, 100, 100, 100])
    D = 2 * np.sqrt(K)
    # q_dd = convex_optimization(model, data, x_t[counter], xd_t[counter], xd, xdd_t[counter],
    # fext, D, K, J, J_dot, Mx_inv, MM, C, dt)
    viewer.sync()
    print("=" * 50)
    # counter += 1
