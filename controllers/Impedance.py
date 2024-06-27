import mujoco
import numpy as np
import time

class Impedance:
    def __init__(self,model,data,viewer) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer

    def setParams(self,Ipos,Iori,Kpos:float,Kori:float,Kp_null,
                  D:float,integration_dt:float,gravity_compensation:bool,
                  dt:float,site_nameL="end_effector",site_nameR="end_effector1"):
        self.Ipos = Ipos
        self.Iori = Iori
        self.Kpos = Kpos
        self.Kori = Kori
        self.Kp_null = Kp_null
        self.D = D
        self.integration_dt = integration_dt
        self.gravity_compensation = gravity_compensation
        self.dt = dt
        self.site_nameL = site_nameL
        self.site_nameR = site_nameR

        self.model.opt.timestep = self.dt

        # Compute damping and stiffness matrices.
        self.damping_pos = self.D * 2 * np.sqrt(self.Ipos)
        self.damping_ori = self.D * 2 * np.sqrt(self.Iori)
        self.Kp = np.concatenate([self.Ipos, self.Iori], axis=0)
        self.Kd = np.concatenate([self.damping_pos, self.damping_ori], axis=0)
        self.Kd_null = self.D * 2 * np.sqrt(self.Kp_null)

        # End-effector site we wish to control.
        self.site_idL = self.model.site(self.site_nameL).id

        self.site_idR = self.model.site(self.site_nameR).id

        # Get the dof and actuator ids for the joints we wish to control. These are copied
        # from the XML file. Feel free to comment out some joints to see the effect on
        # the controller.
        self.joint_names = [self.model.jnt(name).name for name in range(self.model.njnt)]

        self.actuator_names = [self.model.actuator(name).name for name in range(self.model.njnt-1)]


        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])


        self.dof_idsL = self.dof_ids[:9]
        self.dof_idsR = self.dof_ids[9:18]

        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.actuator_names])

        self.actuator_idsL = self.actuator_ids[:9]
        self.actuator_idsR = self.actuator_ids[9:]

        # Initial joint configuration saved as a keyframe in the XML file.
        self.key_name = "home"
        self.key_id = self.model.key(self.key_name).id
        self.q0 = self.model.key(self.key_name).qpos[:18]

        # Mocap body we will control with our mouse.
        self.mocap_nameL = "targetL"
        self.mocap_idL = self.model.body(self.mocap_nameL).mocapid[0]

        self.mocap_nameR = "targetR"
        self.mocap_idR = self.model.body(self.mocap_nameR).mocapid[0]

        # Pre-allocate numpy arrays.
        self.jacR = np.zeros((6, self.model.nv))
        self.jacL = np.zeros((6, self.model.nv))
        self.jac = np.zeros((6, 18)) # the jacobian for the arms

        self.M_all = np.zeros((self.model.nv, self.model.nv))

        self.Mx = np.zeros((6, 6))

        self.eye = np.eye(18)

        self.twistL = np.zeros(6)
        self.twistR = np.zeros(6)

        self.site_quatL = np.zeros(4)
        self.site_quatR = np.zeros(4)

        self.site_quat_conjL = np.zeros(4)
        self.site_quat_conjR = np.zeros(4)

        self.error_quatL = np.zeros(4)
        self.error_quatR = np.zeros(4)


    def resetViewer(self):
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

        # Enable site frame visualization.
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def armCrtl(self):

        step_start = time.time()
        
        # Spatial velocity (aka twist).
        self.dxL = self.data.mocap_pos[self.mocap_idL] - self.data.site(self.site_idL).xpos
        self.twistL[:3] = self.Kpos * self.dxL / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quatL, self.data.site(self.site_idL).xmat)
        mujoco.mju_negQuat(self.site_quat_conjL, self.site_quatL)
        mujoco.mju_mulQuat(self.error_quatL, self.data.mocap_quat[self.mocap_idL], self.site_quat_conjL)
        mujoco.mju_quat2Vel(self.twistL[3:], self.error_quatL, 1.0)
        self.twistL[3:] *= self.Kori / self.integration_dt 

        dxR = self.data.mocap_pos[self.mocap_idR] - self.data.site(self.site_idR).xpos
        self.twistR[:3] = self.Kpos * dxR / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quatR, self.data.site(self.site_idR).xmat)
        mujoco.mju_negQuat(self.site_quat_conjR, self.site_quatR)
        mujoco.mju_mulQuat(self.error_quatR, self.data.mocap_quat[self.mocap_idR], self.site_quat_conjR)
        mujoco.mju_quat2Vel(self.twistR[3:], self.error_quatR, 1.0)
        self.twistR[3:] *= self.Kori / self.integration_dt

        # Jacobian.
        mujoco.mj_jacSite(self.model, self.data, self.jacL[:3], self.jacL[3:], self.site_idL)    
        mujoco.mj_jacSite(self.model, self.data, self.jacR[:3], self.jacR[3:], self.site_idR)

        self.jac[:,:9] = self.jacL[:,:9];
        self.jac[:,9:18] = self.jacR[:,9:18];

        # Compute the task-space inertia matrix.
        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        self.M_inv=self.M_all[:18,:18];
        self.Mx_inv = self.jac @ self.M_inv @ self.jac.T

        if abs(np.linalg.det(self.Mx_inv)) >= 1e-2:
            self.Mx = np.linalg.inv(self.Mx_inv)
        else:
            self.Mx = np.linalg.pinv(self.Mx_inv, rcond=1e-2)
        
        
        # Compute generalized forces.
        self.tau = np.zeros(18)

        self.tau[:9] = self.jac[:,:9].T @ self.Mx @ (self.Kp * self.twistL - self.Kd * (self.jac[:,:9] @ self.data.qvel[self.dof_ids[:9]]))
        self.tau[9:18] = self.jac[:,9:18].T @ self.Mx @ (self.Kp * self.twistR - self.Kd * (self.jac[:,9:18] @ self.data.qvel[self.dof_ids[9:18]]))

        self.Jbar = self.M_inv @ self.jac.T @ self.Mx
        
        self.ddq = self.Kp_null * (self.q0 - self.data.qpos[self.dof_ids[:18]]) - self.Kd_null * self.data.qvel[self.dof_ids[:18]]
        self.tau += (np.eye(self.model.nv-6) - self.jac.T @ self.Jbar.T) @ self.ddq

        # Add gravity compensation.
        if self.gravity_compensation:
            self.tau += self.data.qfrc_bias[self.dof_ids[:18]]

        # Set the control signal and step the simulation.
        self.data.ctrl[self.actuator_ids] = self.tau[self.actuator_ids]
        self.gripperCtrl("open","both")
        time_until_next_step = self.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        return self.tau[self.actuator_ids]
    
    def gripperCtrl(self,state,eef):
        if eef=="both":
            if(state=="open"):
                self.data.ctrl[7:9]=0.04;   #open L gripper
                self.data.ctrl[16:18]=0.04; #open R gripper
            elif(state=="close"):
                self.data.ctrl[7:9]=0.0;   #close L gripper
                self.data.ctrl[16:18]=0.0; #close R gripper
        if eef=="left":
            if(state=="open"):
                self.data.ctrl[7:9]=0.04;   #open L gripper
            elif(state=="close"):
                self.data.ctrl[7:9]=0.0;   #close L gripper
        if eef=="right":
            if(state=="open"):
                self.data.ctrl[16:18]=0.04; #open R gripper
            elif(state=="close"):
                self.data.ctrl[16:18]=0.0; #close R gripper
    