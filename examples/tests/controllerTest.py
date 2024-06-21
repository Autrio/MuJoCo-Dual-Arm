import mujoco
import mujoco.viewer as viewer

import numpy as np
import time


model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";
model = mujoco.MjModel.from_xml_path(model_path);
data = mujoco.MjData(model);

integration_dt:float = 0.1

damping: float = 1e-3

Kpos: float = 100
Kori: float = 100

gravity_compensation: bool = True

dt: float = 0.002

Kn = np.asarray([5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0,1.0,1.0,
                 5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0,1.0,1.0])

max_angvel = 0.785

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt
    model.opt.gravity = 0.00

    # End-effector site we wish to control.
    site_nameL = "end_effector"
    site_idL = model.site(site_nameL).id

    site_nameR = "end_effector1"
    site_idR = model.site(site_nameR).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [model.jnt(name).name for name in range(model.njnt)]
    actuator_names = [model.actuator(name).name for name in range(model.njnt)]


    dof_ids = np.array([model.joint(name).id for name in joint_names])

    dof_idsL = dof_ids[:9]
    dof_idsR = dof_ids[9:]

    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    actuator_idsL = actuator_ids[:9]
    actuator_idsR = actuator_ids[9:]

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body we will control with our mouse.
    mocap_nameL = "targetL"
    mocap_idL = model.body(mocap_nameL).mocapid[0]

    mocap_nameR = "targetR"
    mocap_idR = model.body(mocap_nameR).mocapid[0]

    # Pre-allocate numpy arrays.


    jacR = np.zeros((6, model.nv))
    jacL = np.zeros((6, model.nv))
    jac = np.zeros((6, model.nv))

    diag = damping * np.eye(6)
    eye = np.eye(model.nv)

    twistL = np.zeros(6)
    twistR = np.zeros(6)

    site_quatL = np.zeros(4)
    site_quatR = np.zeros(4)

    site_quat_conjL = np.zeros(4)
    site_quat_conjR = np.zeros(4)

    error_quatL = np.zeros(4)
    error_quatR = np.zeros(4)

       # Define a trajectory for the end-effector site to follow.
    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        # show_left_ui=False,
        # show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # Set the target position of the end-effector site.
            #data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.2, 0.1, 0.1, 0.5)
            
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
            jac[:,9:] = jacR[:,9:];

            dq = np.zeros(18)

            # Damped least squares.
            dq[:9] = jac[:,:9].T @ np.linalg.solve(jac[:,:9] @ jac[:,:9].T + diag, twistL)
            dq[9:] = jac[:,9:].T @ np.linalg.solve(jac[:,9:] @ jac[:,9:].T + diag, twistR)

            # Nullspace control biasing joint velocities towards the home configuration.

            dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()