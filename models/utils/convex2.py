
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
import xml.etree.ElementTree as ET
from tqdm import tqdm

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
    

def params(model,data,dt=0.002,site='tip'):
    q = data.qpos[:]
    J = np.zeros((6,model.nv))
    Jdt = np.zeros((6,model.nv))
    mujoco.mj_jacSite(model,data,J[:3],J[3:],model.site(site).id)
    
    #compute Jdt
    Integration_dt = 2*1e-5
    mujoco.mj_integratePos(model,q,data.qvel,Integration_dt)
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    
    mujoco.mj_jacSite(model,data,Jdt[:3],Jdt[3:],model.site(site).id)
    
    Jdot = (Jdt-J)/Integration_dt

    data.qpos = q
    
    C = data.qfrc_bias  # bias force: Coriolis, centrifugal, gravitational
    
    M_all = np.zeros((model.nv, model.nv))
    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_all, np.eye(model.nv))
    M_inv = M_all[:9, :9]
    # Mx_inv_quat = J[:3] @ M_inv @ J[:3].T
    # Mx_inv_pos = J[3:] @ M_inv @ J[3:].T
    # ic(Mx_inv_quat.shape, Mx_inv_pos.shape)
    # Mx_inv = np.concatenate((Mx_inv_quat, Mx_inv_pos), axis=0)
    # ic(Mx_inv)
    Mx_inv = J @ M_inv @ J.T
  

    mm = np.zeros((model.nv, model.nv))  # mass matrix
    # mm_inv = np.linalg.inv(M_inv)
    mujoco.mj_fullM(model, mm, data.qM)

    fext = np.zeros((6, ))
    # fext[1] = 
    # id = 0
    # mujoco.mj_contactForce(model, data, id, fext)
    # print("External Force : ", fext)
    # print("Joint Force : ", data.qfrc_applied.shape)

    return Mx_inv, mm, C, J, Jdot, fext, q

def optimize(model,data,xr,xrdot,xrddot,J,Jdot,K,D,Mx_inv,MM,C,counter):
    qd = data.qvel[:]
    xpos = data.site("tip").xpos

    Temp = data.site("tip").xmat
    Xquat = np.zeros((4,))
    mujoco.mju_mat2Quat(Xquat,Temp)
    
    xori = Quat2rot(Xquat,"wxyz","xyz",False)
    x = np.concatenate((xpos,xori))
    
    xdot = np.zeros((6,))
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip"), xdot, 0)
    
    e = xr - x
    edot = xrdot - xdot
    
    fq = data.qfrc_passive
    fext = J @ fq
    
    xddot = xrddot + Mx_inv @ (D @ edot + K @ e - fext)
    
    qddot = cp.Variable(9)
    
    tau_max = np.array([1000] * 9)
    tau_min = np.array([-1000] * 9)
    qddot_max = np.array([1000] * 9)
    qddot_min = np.array([-1000] * 9)
    
    objective = cp.Minimize(1/2*cp.sum_squares(J @ qddot + Jdot @ qd - xddot))
    constraints = [MM @ qddot + C <= tau_max,
                     MM @ qddot + C >= tau_min,
                     qddot <= qddot_max,
                     qddot >= qddot_min]
    
    problem = cp.Problem(objective,constraints[:4])
    
    try:
        loss = problem.solve(verbose=True, max_iter=100000)
    except:
        return None
    ic(qddot.value)
    if(loss > 5):
        print("=========================")
        print("Loss: ", loss)
        print("time: ",counter)
        print("=========================")
    ic(loss)
    if(loss > 1):
        exit()
    print("end"*10)
    # ic(q_dd.value)
    # ic(constraints[0].dual_value)
    return qddot.value


def mujoco2normal(q):
    return np.array([q[1], q[2], q[3], q[0]])

def normal2mujoco(q):
    return np.array([q[3], q[0], q[1], q[2]])


def slerp(x_init_quat, x_final_quat, final_n, n_steps):
    X_init_quat = mujoco2normal(x_init_quat)
    X_final_quat = mujoco2normal(x_final_quat)
    
    rot_times = np.array([0, final_n])
    rots = R.from_quat([X_init_quat, X_final_quat])
    
    slerp = Slerp(rot_times, rots)
    times = np.linspace(0, final_n, n_steps)
    rpy = slerp(times).as_euler('xyz', degrees=False)    
    
    rpy_dot = np.diff(rpy, axis=0) / (final_n / n_steps) # 499, 3
    rpy_ddot = np.diff(rpy_dot, axis=0) / (final_n / n_steps) # 498, 3
    
    rpy_dot_f, rpy_ddot_f = np.zeros_like(rpy), np.zeros_like(rpy)
    rpy_dot_f[:-1, :] = copy.deepcopy(rpy_dot)
    rpy_dot_f[-1:, :] = copy.deepcopy(rpy_dot[-1, :])
    rpy_ddot_f[:-2, :] = copy.deepcopy(rpy_ddot)
    rpy_ddot_f[-2:, :] = copy.deepcopy(rpy_ddot[-2, :])
    
    return rpy, rpy_dot_f, rpy_ddot_f

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
    #! NOTE: need to fix this
    # model.opt.gravity = -9.81
    model.opt.gravity = 0
    
    #* first put the robot in home position
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    viewer.sync()
    
    dt = model.opt.timestep

    force_site = model.site('tip').id

    end_effector_mat = data.site('tip').xmat
    end_effector_quat = np.zeros((4, ))
    mujoco.mju_mat2Quat(end_effector_quat, end_effector_mat)
    final_end_effector_quat = np.zeros((4, ))
    final_end_effector_quat[1] = 1
    # final_end_effector_quat[1] = np.sqrt(1/2)
    # final_end_effector_quat[2] = np.sqrt(1/2)
    

    x_init = data.site(force_site).xpos.reshape(-1, 1)
    x_final = copy.deepcopy(x_init) + np.array([0.0, 0.2, -0.3]).reshape(-1, 1)
    
    # ic(end_effector_quat)
    # exit()
    timesteps = 2000
    
    quat_ref, quat_dot_ref, quat_ddot_ref = slerp(end_effector_quat, final_end_effector_quat, timesteps, timesteps)
    pos_ref, pos_dot_ref, pos_ddot_ref = quintic_pos(x_init, x_final, timesteps, timesteps)

    pose_ref = np.concatenate([pos_ref, quat_ref], axis=-1) # 500, 7
    pose_dot_ref = np.concatenate([pos_dot_ref, quat_dot_ref], axis=-1) # 500, 7
    pose_ddot_ref = np.concatenate([pos_ddot_ref, quat_ddot_ref], axis=-1) # 500, 7
    
    np.save("./pose_ref.npy", pose_ref)

    # ic(pose_ref.shape, pose_dot_ref.shape, pose_ddot_ref.shape)

    counter = 0

    K = np.diag([1000.0] * 6)
    D = 2 * np.sqrt(K)
    
    pbar = tqdm(total=timesteps, colour='blue')
    
    pause_sim = False


    while viewer.is_running():
        data.mocap_pos[model.body("target").mocapid[0]] = x_final.reshape(-1)

        Mx_inv, MM, C, J, Jdot, fext, q = params(model, data,dt)
        
        if not pause_sim and counter < timesteps:
            qddot = optimize(model,data,pose_ref[counter],pose_dot_ref[counter],pose_ddot_ref[counter],J,Jdot,K,D,Mx_inv,MM,C,counter)
            
            if qddot is not None:
                optTau = np.zeros((9,))
                optTau = MM @ qddot + C
                data.ctrl[:9] = optTau
                
                mujoco.mj_step(model, data)
            # pbar.update(1)  
        
        viewer.sync()
    
        # print("=" * 50)
        counter += 1
        
if __name__=="__main__":
    main()