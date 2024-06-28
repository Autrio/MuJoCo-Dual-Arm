import mujoco
import mujoco.viewer as viewer

import numpy as np
import time
import argparse as ap

parser = ap.ArgumentParser(prog="OpSpcTest",
                           description="Testing code to verify operational space control for two franka panda arms")

parser.add_argument("-d","--model",type=str,help="""Choose variant of dual panda arms.
                    'dual' for individual separate arms, 'bimanual' for arms connected
                     to a torso at shoulder joint. Default is 'dual'""")

args = parser.parse_args()


if(args.model == "bimanual"):
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/bimanual_panda.xml";
else:
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";
model = mujoco.MjModel.from_xml_path(model_path);
data = mujoco.MjData(model);


# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([200.0, 200.0, 200.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([50.0, 50.0, 25.0, 25.0, 12.5, 5.0, 2.0, 2.0, 2.0,
                      50.0, 50.0, 25.0, 25.0, 12.5, 5.0, 2.0, 2.0, 2.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.3

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 1

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 1

# Integration timestep in seconds.
integration_dt: float = 0.1

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

def gripperCtrl(state,eef):
    if eef=="both":
        if(state=="open"):
            data.ctrl[7:9]=0.04;   #open L gripper
            data.ctrl[16:18]=0.04; #open R gripper
        elif(state=="close"):
            data.ctrl[7:9]=0.0;   #close L gripper
            data.ctrl[16:18]=0.0; #close R gripper
    if eef=="left":
        if(state=="open"):
            data.ctrl[7:9]=0.04;   #open L gripper
        elif(state=="close"):
            data.ctrl[7:9]=0.0;   #close L gripper
    if eef=="right":
        if(state=="open"):
            data.ctrl[16:18]=0.04; #open R gripper
        elif(state=="close"):
            data.ctrl[16:18]=0.0; #close R gripper        


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    model.opt.timestep = dt

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_nameL = "end_effector"
    site_idL = model.site(site_nameL).id

    site_nameR = "end_effector1"
    site_idR = model.site(site_nameR).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [model.jnt(name).name for name in range(model.njnt)]

    actuator_names = [model.actuator(name).name for name in range(model.njnt-1)]


    dof_ids = np.array([model.joint(name).id for name in joint_names])


    dof_idsL = dof_ids[:9]
    dof_idsR = dof_ids[9:18]

    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    actuator_idsL = actuator_ids[:9]
    actuator_idsR = actuator_ids[9:]

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos[:18]

    # Mocap body we will control with our mouse.
    mocap_nameL = "targetL"
    mocap_idL = model.body(mocap_nameL).mocapid[0]

    mocap_nameR = "targetR"
    mocap_idR = model.body(mocap_nameR).mocapid[0]

    # Pre-allocate numpy arrays.


    jacR = np.zeros((6, model.nv))
    jacL = np.zeros((6, model.nv))
    jac = np.zeros((6, 18)) # the jacobian for the arms

    M_all = np.zeros((model.nv, model.nv))

    Mx = np.zeros((6, 6))

    eye = np.eye(18)

    twistL = np.zeros(6)
    twistR = np.zeros(6)

    site_quatL = np.zeros(4)
    site_quatR = np.zeros(4)

    site_quat_conjL = np.zeros(4)
    site_quat_conjR = np.zeros(4)

    error_quatL = np.zeros(4)
    error_quatR = np.zeros(4)


    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            
            # Spatial velocity (aka twist).
            dxL = data.mocap_pos[mocap_idL] - data.site(site_idL).xpos
            twistL[:3] = Kpos * dxL / integration_dt
            mujoco.mju_mat2Quat(site_quatL, data.site(site_idL).xmat)
            mujoco.mju_negQuat(site_quat_conjL, site_quatL)
            mujoco.mju_mulQuat(error_quatL, data.mocap_quat[mocap_idL], site_quat_conjL)
            mujoco.mju_quat2Vel(twistL[3:], error_quatL, 1.0)
            twistL[3:] *= Kori / integration_dt 

            dxR = data.mocap_pos[mocap_idR] - data.site(site_idR).xpos
            twistR[:3] = Kpos * dxR / integration_dt
            mujoco.mju_mat2Quat(site_quatR, data.site(site_idR).xmat)
            mujoco.mju_negQuat(site_quat_conjR, site_quatR)
            mujoco.mju_mulQuat(error_quatR, data.mocap_quat[mocap_idR], site_quat_conjR)
            mujoco.mju_quat2Vel(twistR[3:], error_quatR, 1.0)
            twistR[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jacL[:3], jacL[3:], site_idL)    
            mujoco.mj_jacSite(model, data, jacR[:3], jacR[3:], site_idR)

            jac[:,:9] = jacL[:,:9];
            jac[:,9:18] = jacR[:,9:18];

            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_all, np.eye(model.nv))
            M_inv=M_all[:18,:18];
            Mx_inv = jac @ M_inv @ jac.T

            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
            
            
            # Compute generalized forces.
            tau = np.zeros(18)

            tau[:9] = jac[:,:9].T @ Mx @ (Kp * twistL - Kd * (jac[:,:9] @ data.qvel[dof_ids[:9]]))
            tau[9:18] = jac[:,9:18].T @ Mx @ (Kp * twistR - Kd * (jac[:,9:18] @ data.qvel[dof_ids[9:18]]))


            Jbar = M_inv @ jac.T @ Mx
            
            ddq = Kp_null * (q0 - data.qpos[dof_ids[:18]]) - Kd_null * data.qvel[dof_ids[:18]]
            tau += (np.eye(model.nv-6) - jac.T @ Jbar.T) @ ddq

            # Add gravity compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids[:18]]

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = tau[actuator_ids]
            gripperCtrl("open","both")
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print(data.mocap_pos[mocap_idL],data.mocap_quat[mocap_idL])
    print(data.mocap_pos[mocap_idR],data.mocap_quat[mocap_idR])


if __name__ == "__main__":
    main()