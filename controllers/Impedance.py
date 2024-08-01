import mujoco
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


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
        self.qd0 = self.data.qvel[:18]

        # Mocap body we will control with our mouse.
        self.mocap_nameL = "targetL"
        self.mocap_idL = self.model.body(self.mocap_nameL).mocapid[0]

        self.mocap_nameR = "targetR"
        self.mocap_idR = self.model.body(self.mocap_nameR).mocapid[0]

        # Pre-allocate numpy arrays.
        self.jacR = np.zeros((6, self.model.nv))
        self.jacL = np.zeros((6, self.model.nv))
        self.jac = np.zeros((6, 18)) # the jacobian for the arms
        self.jacPrev = np.zeros((6,18)) #prev values of jac for finite difference jdot

    
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

        #arrays for plotting
        self.setupDatacap()

    def resetViewer(self,flag):
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

        # Enable site frame visualization.
        if(flag):
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE    

    def armCtrl(self,JacP):

        step_start = time.time()
        
        self.xL = self.data.site(self.site_idL).xpos
        self.xR = self.data.site(self.site_idR).xpos
        
        # Spatial velocity (aka twist).
        self.dxL = self.data.mocap_pos[self.mocap_idL] - self.data.site(self.site_idL).xpos
        self.twistL[:3] = self.Kpos * self.dxL / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quatL, self.data.site(self.site_idL).xmat)
        mujoco.mju_negQuat(self.site_quat_conjL, self.site_quatL)
        mujoco.mju_mulQuat(self.error_quatL, self.data.mocap_quat[self.mocap_idL], self.site_quat_conjL)
        mujoco.mju_quat2Vel(self.twistL[3:], self.error_quatL, 1.0)
        self.twistL[3:] *= self.Kori / self.integration_dt 

        self.dxR = self.data.mocap_pos[self.mocap_idR] - self.data.site(self.site_idR).xpos
        self.twistR[:3] = self.Kpos * self.dxR / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quatR, self.data.site(self.site_idR).xmat)
        mujoco.mju_negQuat(self.site_quat_conjR, self.site_quatR)
        mujoco.mju_mulQuat(self.error_quatR, self.data.mocap_quat[self.mocap_idR], self.site_quat_conjR)
        mujoco.mju_quat2Vel(self.twistR[3:], self.error_quatR, 1.0)
        self.twistR[3:] *= self.Kori / self.integration_dt


        # implement as a function later
        tempQuatL = self.error_quatL[1:]
        tempQuatL = np.append(tempQuatL,self.error_quatL[0])
        rotnErrL = R.from_quat(tempQuatL)
        self.roL = rotnErrL.as_euler("xyz",degrees=False)

        tempQuatR = self.error_quatR[1:]
        tempQuatR = np.append(tempQuatR,self.error_quatR[0])
        rotnErrR = R.from_quat(tempQuatR)
        self.roR = rotnErrR.as_euler("xyz",degrees=False)

        # Jacobian.
        mujoco.mj_jacSite(self.model, self.data, self.jacL[:3], self.jacL[3:], self.site_idL)    
        mujoco.mj_jacSite(self.model, self.data, self.jacR[:3], self.jacR[3:], self.site_idR)

        self.jac[:,:9] = self.jacL[:,:9];
        self.jac[:,9:18] = self.jacR[:,9:18];

        self.M = np.zeros((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model,self.M,self.data.qM)
        self.M = self.M[:18,:18]

        # Compute the task-space inertia matrix.
        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        self.M_inv=self.M_all[:18,:18];
        self.Mx_inv = self.jac @ self.M_inv @ self.jac.T

        if abs(np.linalg.det(self.Mx_inv)) >= 1e-2:
            self.Mx = np.linalg.inv(self.Mx_inv)
        else:
            self.Mx = np.linalg.pinv(self.Mx_inv, rcond=1e-2)

        #compute H(q,qdot)
        self.jacPrev = JacP        
        self.Jdot = (self.jac - self.jacPrev)/self.integration_dt
        self.h = self.data.qfrc_bias[self.dof_ids[:18]]
        self.mu = self.Mx @ (self.jac @ self.M_inv @ self.h + self.Jdot @ self.data.qvel[:18])


        # Compute generalized forces.
        self.tau = np.zeros(18)

        self.tau[:9] = self.jac[:,:9].T @ (self.Kd * np.concatenate((self.dxL ,self.roL)) + self.Kp * self.twistL +  self.mu)
        self.tau[9:18] = self.jac[:,9:18].T @ (self.Kd * np.concatenate((self.dxR ,self.roR)) + self.Kp * self.twistR +  self.mu)

        #compute postural constraints
        self.Jbar = self.M_inv @ self.jac.T @ self.Mx
        self.ddq = self.Kp_null * (self.q0 - self.data.qpos[self.dof_ids[:18]]) - self.Kd_null * (self.qd0 - self.data.qvel[self.dof_ids[:18]])
        self.tau1 = self.M @ self.ddq + self.h
        self.tau += (np.eye(self.model.nv-6) - self.jac.T @ self.Jbar.T) @ self.tau1

        # Set the control signal and step the simulation.
        self.data.ctrl[self.actuator_ids] = self.tau[self.actuator_ids]
        self.gripperCtrl("open","both")
        time_until_next_step = self.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # Collect data for plotting
        self.Datacap()

        #return the actuator torques for monitoring
        return self.tau[self.actuator_ids]
    
    def gripperCtrl(self,state,eef):
        if eef=="both":
            if(state=="open"):
                self.data.ctrl[7:9]=10;   #open L gripper
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
    
    def setupDatacap(self):
        self.time_steps = []
        self.dxL_xlist = []
        self.dxL_ylist = []
        self.dxL_zlist = []
        self.dxR_xlist = []
        self.dxR_ylist = []
        self.dxR_zlist = []
        self.error_quatL_xlist = []
        self.error_quatL_ylist = []
        self.error_quatL_zlist = []
        self.error_quatL_wlist = []
        self.error_quatR_xlist = []
        self.error_quatR_ylist = []
        self.error_quatR_zlist = []
        self.error_quatR_wlist = []
        self.erL_list = []
        self.erR_list = []
        self.er_quatL_list = []
        self.er_quatR_list = []
        self.posL_list = []
        self.posR_list = []
        self.quatL_list = []
        self.quatR_list = []
        self.erL_joint = []
        self.erR_joint = []
        self.joint_trajL = []
        self.joint_trajR = []
        self.forceL_list = []
        self.forceR_list = []
        self.object_pos = []
        self.object_quat = []
        self.SD = {}
        self.SD["left_wrist_force"] = []
        self.SD["right_wrist_force"] = []
        self.SD["left_finger1_force"] = []
        self.SD["left_finger2_force"] = []
        self.SD["right_finger1_force"] = []
        self.SD["right_finger2_force"] = []
        self.iter = 0

        
    def Datacap(self):
        self.erL = np.linalg.norm(np.append(self.dxL,self.roL))
        self.erR = np.linalg.norm(np.append(self.dxR,self.roR))
        self.time_steps.append(len(self.time_steps) * self.dt)
        self.dxL_xlist.append(self.dxL[0])
        self.dxL_ylist.append(self.dxL[1])
        self.dxL_zlist.append(self.dxL[2])
        self.dxR_xlist.append(self.dxR[0])
        self.dxR_ylist.append(self.dxR[1])
        self.dxR_zlist.append(self.dxR[2])
        self.error_quatL_xlist.append(self.error_quatL[0])
        self.error_quatL_ylist.append(self.error_quatL[1])
        self.error_quatL_zlist.append(self.error_quatL[2])
        self.error_quatL_wlist.append(self.error_quatL[3])
        self.error_quatR_xlist.append(self.error_quatR[0])
        self.error_quatR_ylist.append(self.error_quatR[1])
        self.error_quatR_zlist.append(self.error_quatR[2])
        self.error_quatR_wlist.append(self.error_quatR[3])
        self.erL_list.append(self.erL)
        self.erR_list.append(self.erR)
        self.er_quatL_list.append(np.linalg.norm(self.error_quatL))
        self.er_quatR_list.append(np.linalg.norm(self.error_quatR))
        self.posL_list.append(self.data.mocap_pos[self.mocap_idL].copy())
        self.posR_list.append(self.data.mocap_pos[self.mocap_idR].copy())
        self.quatL_list.append(self.data.mocap_quat[self.mocap_idL].copy())
        self.quatR_list.append(self.data.mocap_quat[self.mocap_idR].copy())
        self.joint_trajL.append(self.data.qpos[self.dof_ids[:9]].copy())
        self.joint_trajR.append(self.data.qpos[self.dof_ids[9:18]].copy())
        self.forceL_list.append(np.linalg.norm(self.tau[8]))  # Example of force collection
        self.forceR_list.append(np.linalg.norm(self.tau[17]))  # Example of force collection
        self.object_pos.append(self.data.body("collision_object").xpos.copy())
        self.object_quat.append(self.data.body("collision_object").xquat.copy())
        
        self.SD["left_wrist_force"].insert(-1,self.data.sensor("LAjaf7").data.copy())
        self.SD["right_wrist_force"].append(self.data.sensor("RAjaf7").data.copy())
        self.SD["left_finger1_force"].append(self.data.sensor("LHjafF1").data.copy())
        self.SD["left_finger2_force"].append(self.data.sensor("LHjafF2").data.copy())
        self.SD["right_finger1_force"].append(self.data.sensor("RHjafF1").data.copy())
        self.SD["right_finger2_force"].append(self.data.sensor("RHjafF2").data.copy())        

    def makeplots(self):
        # Convert lists to numpy arrays for easier manipulation
        self.time_steps = np.array(self.time_steps)
        self.dxL_xlist = np.array(self.dxL_xlist)
        self.dxL_ylist = np.array(self.dxL_ylist)
        self.dxL_zlist = np.array(self.dxL_zlist)
        self.dxR_xlist = np.array(self.dxR_xlist)
        self.dxR_ylist = np.array(self.dxR_ylist)
        self.dxR_zlist = np.array(self.dxR_zlist)
        self.error_quatL_xlist = np.array(self.error_quatL_xlist)
        self.error_quatL_ylist = np.array(self.error_quatL_ylist)
        self.error_quatL_zlist = np.array(self.error_quatL_zlist)
        self.error_quatL_wlist = np.array(self.error_quatL_wlist)
        self.error_quatR_xlist = np.array(self.error_quatR_xlist)
        self.error_quatR_ylist = np.array(self.error_quatR_ylist)
        self.error_quatR_zlist = np.array(self.error_quatR_zlist)
        self.error_quatR_wlist = np.array(self.error_quatR_wlist)        
        self.erL_list = np.array(self.erL_list)
        self.erR_list = np.array(self.erR_list)
        self.er_quatL_list = np.array(self.er_quatL_list)
        self.er_quatR_list = np.array(self.er_quatR_list)
        self.posL_list = np.array(self.posL_list)
        self.posR_list = np.array(self.posR_list)
        self.quatL_list = np.array(self.quatL_list)
        self.quatR_list = np.array(self.quatR_list)
        self.erL_joint = np.array(self.erL_joint)
        self.erR_joint = np.array(self.erR_joint)
        self.joint_trajL = np.array(self.joint_trajL)
        self.joint_trajR = np.array(self.joint_trajR)
        self.forceL_list = np.array(self.forceL_list)
        self.forceR_list = np.array(self.forceR_list)
        self.object_pos = np.array(self.object_pos)
        self.object_quat = np.array(self.object_quat)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.plot(self.time_steps, self.posL_list[:, 0], label='x')
        plt.plot(self.time_steps, self.posL_list[:, 1], label='y')
        plt.plot(self.time_steps, self.posL_list[:, 2], label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Left Arm End-Effector Position in Cartesian Space')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.time_steps, self.posR_list[:, 0], label='x')
        plt.plot(self.time_steps, self.posR_list[:, 1], label='y')
        plt.plot(self.time_steps, self.posR_list[:, 2], label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Right Arm End-Effector Position in Cartesian Space')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.time_steps, self.quatL_list[:, 0], label='x')
        plt.plot(self.time_steps, self.quatL_list[:, 1], label='y')
        plt.plot(self.time_steps, self.quatL_list[:, 2], label='z')
        plt.plot(self.time_steps, self.quatL_list[:, 3], label='w')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion')
        plt.title('Left Arm End-Effector Orientation in Quaternion Space')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.time_steps, self.quatR_list[:, 0], label='x')
        plt.plot(self.time_steps, self.quatR_list[:, 1], label='y')
        plt.plot(self.time_steps, self.quatR_list[:, 2], label='z')
        plt.plot(self.time_steps, self.quatR_list[:, 3], label='w')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion')
        plt.title('Right Arm End-Effector Orientation in Quaternion Space')
        plt.legend()

        plt.tight_layout()

        # Plot for errors in Cartesian space for both arms --------------------------------------------------------
        plt.figure(figsize=(12, 6))

        plt.subplot(4, 2, 1)
        plt.plot(self.time_steps, self.dxL_xlist, label='x')
        plt.plot(self.time_steps, self.dxL_ylist, label='y')
        plt.plot(self.time_steps, self.dxL_zlist, label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Left Arm Position Errors in Cartesian Space')
        plt.legend()

        plt.subplot(4, 2, 2)
        plt.plot(self.time_steps, self.dxR_xlist, label='x')
        plt.plot(self.time_steps, self.dxR_ylist, label='y')
        plt.plot(self.time_steps, self.dxR_zlist, label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Right Arm Position Errors in Cartesian Space')
        plt.legend()

        plt.subplot(4, 2, 3)
        plt.plot(self.time_steps, self.error_quatL_xlist, label='x')
        plt.plot(self.time_steps, self.error_quatL_ylist, label='y')
        plt.plot(self.time_steps, self.error_quatL_zlist, label='z')
        plt.plot(self.time_steps, self.error_quatL_wlist, label='w')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion Error')
        plt.title('Left Arm Orientation Error in Quaternion Space')
        plt.legend()

        plt.subplot(4, 2, 4)
        plt.plot(self.time_steps, self.error_quatR_xlist, label='x')
        plt.plot(self.time_steps, self.error_quatR_ylist, label='y')
        plt.plot(self.time_steps, self.error_quatR_zlist, label='z')
        plt.plot(self.time_steps, self.error_quatR_wlist, label='w')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion Error')
        plt.title('Right Arm Orientation Error in Quaternion Space')
        plt.legend()


        plt.subplot(4, 2, 5)
        plt.plot(self.time_steps, self.erL_list, label='Left Arm Position Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Left Arm Position Error (Norm) in Cartesian Space')
        plt.legend()

        plt.subplot(4, 2, 6)
        plt.plot(self.time_steps, self.erR_list, label='Right Arm Position Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Right Arm Position Error (Norm) in Cartesian Space')
        plt.legend()

        plt.subplot(4, 2, 7)
        plt.plot(self.time_steps, self.er_quatL_list, label='Left Arm Orientation Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Orientation Error')
        plt.title('Left Arm Orientation Error (Norm) in Quaternion Space')
        plt.legend()

        plt.subplot(4, 2, 8)
        plt.plot(self.time_steps, self.er_quatR_list, label='Right Arm Orientation Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Orientation Error')
        plt.title('Right Arm Orientation Error (Norm) in Quaternion Space')
        plt.legend()
               
        plt.tight_layout()

        # Plot for joint trajectories -----------------------------------------------------------------------------
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.time_steps, self.joint_trajL[:, 0], label='Joint 1')
        plt.plot(self.time_steps, self.joint_trajL[:, 1], label='Joint 2')
        plt.plot(self.time_steps, self.joint_trajL[:, 2], label='Joint 3')
        plt.plot(self.time_steps, self.joint_trajL[:, 3], label='Joint 4')
        plt.plot(self.time_steps, self.joint_trajL[:, 4], label='Joint 5')
        plt.plot(self.time_steps, self.joint_trajL[:, 5], label='Joint 6')
        plt.plot(self.time_steps, self.joint_trajL[:, 6], label='Joint 7')
        plt.plot(self.time_steps, self.joint_trajL[:, 7], label='Left_gripper')
        plt.plot(self.time_steps, self.joint_trajL[:, 8], label='Right_gripper')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angles (rad)')
        plt.title('Joint Angles for Left Arm')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.time_steps, self.joint_trajR[:, 0], label='Joint 1')
        plt.plot(self.time_steps, self.joint_trajR[:, 1], label='Joint 2')
        plt.plot(self.time_steps, self.joint_trajR[:, 2], label='Joint 3')
        plt.plot(self.time_steps, self.joint_trajR[:, 3], label='Joint 4')
        plt.plot(self.time_steps, self.joint_trajR[:, 4], label='Joint 5')
        plt.plot(self.time_steps, self.joint_trajR[:, 5], label='Joint 6')
        plt.plot(self.time_steps, self.joint_trajR[:, 6], label='Joint 7')
        plt.plot(self.time_steps, self.joint_trajR[:, 7], label='Left_gripper')
        plt.plot(self.time_steps, self.joint_trajR[:, 8], label='Right_gripper')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angles (rad)')
        plt.title('Joint Angles for Right Arm')
        plt.legend()

        plt.tight_layout()

        #plot for object trajectories ----------------------------------------------------------------------
        plt.figure(figsize=(12,6))
        plt.subplot(2,2,1)
        plt.plot(self.time_steps, self.object_pos[:,0], label='x')
        plt.plot(self.time_steps, self.object_pos[:,1], label='y')
        plt.plot(self.time_steps, self.object_pos[:,2], label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Object Position')
        plt.title('Object Position')
        plt.legend()

        plt.subplot(2,2,2)
        plt.plot(self.time_steps, self.object_quat[:,0], label='w')
        plt.plot(self.time_steps, self.object_quat[:,1], label='x')
        plt.plot(self.time_steps, self.object_quat[:,2], label='y')
        plt.plot(self.time_steps, self.object_quat[:,3], label='z')
        plt.xlabel('Time (s)')
        plt.ylabel('Object Rotation')
        plt.title('Object Rotation')
        plt.legend()

        #sensor data ---------------------------------------------------------------------------------------

        plt.figure(figsize=(12,6))

        sensorid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'LAjaf7')
        
        plt.subplot(2,2,1)
        plt.plot(range(len(self.SD["left_wrist_force"])),self.SD["left_wrist_force"][::-1],label="left joint7 force")
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Left joint7 Force')
        plt.legend()


        plt.subplot(2,2,2)
        plt.plot(range(len(self.SD["right_wrist_force"])),self.SD["right_wrist_force"][::-1],label="right joint7 force")
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Right joint7 Force')
        plt.legend()
        

        plt.subplot(2,2,3)
        plt.plot(range(len(self.SD["left_finger1_force"])),self.SD["left_finger1_force"][::-1],label="left finger1 force")
        plt.plot(range(len(self.SD["left_finger2_force"])),self.SD["left_finger2_force"][::-1],label="left finger2 force")
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Left hand finger Force')
        plt.legend()


        plt.subplot(2,2,4)
        plt.plot(range(len(self.SD["right_finger1_force"])),self.SD["right_finger1_force"][::-1],label="right finger1 force")
        plt.plot(range(len(self.SD["right_finger2_force"])),self.SD["right_finger2_force"][::-1],label="right finger2 force")
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Right hand finger Force')
        plt.legend()

        plt.tight_layout()
        plt.show()
