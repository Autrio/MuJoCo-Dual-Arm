import cvxpy 
import mujoco 

import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from controller.QuinticPolynomial import create_quintic_trajectory, QuinticPolynomial

class Convex:
    def __init__(self,model,data,viewer):
        self.model = model
        self.data = data
        self.viewer = viewer

    def resetViewer(self,flag):
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

        # Enable site frame visualization.
        if(flag):
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE    
            
    def safe_matrix_sqrt(matrix):
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Take square root of the absolute values of the eigenvalues
        sqrt_eigvals = np.sqrt(np.abs(eigvals))
        
        # Reconstruct the square root of the matrix
        sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
        
        return sqrt_matrix

    def compute_damping_matrices(MxL, MxR, K):
        # Compute the square roots using the safe square root function
        sqrt_MxL = safe_matrix_sqrt(MxL)
        sqrt_MxR = safe_matrix_sqrt(MxR)
        sqrt_K = np.sqrt(K)  # K has no negative eigenvalues
        
        # Compute the damping matrices
        DL = sqrt_MxL @ sqrt_K + sqrt_K @ sqrt_MxL
        DR = sqrt_MxR @ sqrt_K + sqrt_K @ sqrt_MxR
        
        return DL, DR

    def SetStaticParams(self,K,K_null,Kpos,Kori,dt,D=None,site_nameL="end_effector",site_nameR="end_effector1"):

        self.K = np.diag(K)
        # self.D = np.diag(D)
        self.Kpos = Kpos
        self.Kori = Kori
        self.K_null = np.diag(K_null)
        self.site_nameL = site_nameL
        self.site_nameR = site_nameR
        self.dt = dt

        self.joint_names = [self.model.jnt(name).name for name in range(self.model.njnt)]

        self.actuator_names = [self.model.actuator(name).name for name in range(self.model.njnt-1)]


        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])


        self.dof_idsL = self.dof_ids[:9]
        self.dof_idsR = self.dof_ids[9:18]

        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.actuator_names])

        self.actuator_idsL = self.actuator_ids[:9]
        self.actuator_idsR = self.actuator_ids[9:]

        self.site_idL = self.model.site(self.site_nameL).id

        self.site_idR = self.model.site(self.site_nameR).id

        self.key_name = "home"
        self.key_id = self.model.key(self.key_name).id
        self.q0 = self.model.key(self.key_name).qpos[:18]
        self.qd0 = self.data.qvel[:18]

        # Mocap body we will control with our mouse.
        self.mocap_nameL = "targetL"
        self.mocap_idL = self.model.body(self.mocap_nameL).mocapid[0]

        self.mocap_nameR = "targetR"
        self.mocap_idR = self.model.body(self.mocap_nameR).mocapid[0]

        self.jacR = np.zeros((6, self.model.nv))
        self.jacL = np.zeros((6, self.model.nv))
        self.jacPrevL = np.zeros((6,9)) # the jacobian for the arms
        self.jacPrevR = np.zeros((6,9)) #prev values of jac for finite difference jdot

        self.JR = np.zeros((6,9))
        self.JL = np.zeros((6,9))


    
        self.M_all = np.zeros((self.model.nv, self.model.nv)) # Will be populated with Minv Values

        self.MxL = np.zeros((6, 6))  #task space inertia martix
        self.MxR = np.zeros((6, 6))  #task space inertia martix


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
        # self.setupDatacap()
    

    
    def SetDynamicParams(self,PosturalBias,velocityBias,JacPL,JacPR):

        self.xL = self.data.site(self.site_idL).xpos
        self.xR = self.data.site(self.site_idR).xpos
        
        # Spatial velocity (aka twist).
        self.dxL = self.data.mocap_pos[self.mocap_idL] - self.data.site(self.site_idL).xpos
        self.twistL[:3] = self.Kpos * self.dxL / self.dt
        mujoco.mju_mat2Quat(self.site_quatL, self.data.site(self.site_idL).xmat)
        mujoco.mju_negQuat(self.site_quat_conjL, self.site_quatL)
        mujoco.mju_mulQuat(self.error_quatL, self.data.mocap_quat[self.mocap_idL], self.site_quat_conjL)
        mujoco.mju_quat2Vel(self.twistL[3:], self.error_quatL, 1.0)
        # self.twistL[3:] *= self.Kori / self.dt 

        self.dxR = self.data.mocap_pos[self.mocap_idR] - self.data.site(self.site_idR).xpos
        self.twistR[:3] = self.Kpos* self.dxR / self.dt
        mujoco.mju_mat2Quat(self.site_quatR, self.data.site(self.site_idR).xmat)
        mujoco.mju_negQuat(self.site_quat_conjR, self.site_quatR)
        mujoco.mju_mulQuat(self.error_quatR, self.data.mocap_quat[self.mocap_idR], self.site_quat_conjR)
        mujoco.mju_quat2Vel(self.twistR[3:], self.error_quatR, 1.0)
        # self.twistR[3:] *= self.Kori / self.dt


        # implement as a function later
        self.tempQuatL = self.error_quatL[1:]
        self.tempQuatL = np.append(self.tempQuatL,self.error_quatL[0])
        self.rotnErrL = R.from_quat(self.tempQuatL)
        self.roL = self.rotnErrL.as_euler("xyz",degrees=False)

        self.tempQuatR = self.error_quatR[1:]
        self.tempQuatR = np.append(self.tempQuatR,self.error_quatR[0])
        self.rotnErrR = R.from_quat(self.tempQuatR)
        self.roR = self.rotnErrR.as_euler("xyz",degrees=False)

        self.PosErrL = np.concatenate((self.dxL,self.roL))
        self.PosErrR = np.concatenate((self.dxR,self.roR))


        # Jacobian.
        mujoco.mj_jacSite(self.model, self.data, self.jacL[:3], self.jacL[3:], self.site_idL)    
        mujoco.mj_jacSite(self.model, self.data, self.jacR[:3], self.jacR[3:], self.site_idR)

        self.JL = self.jacL[:,:9];
        self.JR = self.jacR[:,9:18];

        self.M = np.zeros((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model,self.M,self.data.qM)
        self.ML = self.M[:9,:9]
        self.MR = self.M[9:18,9:18]


        # Compute the task-space inertia matrix.
        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        self.ML_inv=self.M_all[:9,:9];
        self.MR_inv=self.M_all[9:18,9:18];

        self.MxL_inv = self.JL @ self.ML_inv @ self.JL.T
        self.MxR_inv = self.JR @ self.MR_inv @ self.JR.T


        if abs(np.linalg.det(self.MxL_inv)) >= 1e-2:
            self.MxL = np.linalg.inv(self.MxL_inv)
        else:
            self.MxL = np.linalg.pinv(self.MxL_inv, rcond=1e-2)

        if abs(np.linalg.det(self.MxR_inv)) >= 1e-2:
            self.MxR = np.linalg.inv(self.MxR_inv)
        else:
            self.MxR = np.linalg.pinv(self.MxR_inv, rcond=1e-2)



        #compute H(q,qdot)
        self.jacLPrev = JacPL        
        self.JLdot = (self.JL - self.jacLPrev)/self.dt

        self.jacRPrev = JacPR        
        self.JRdot = (self.JR - self.jacRPrev)/self.dt

        self.hL = self.data.qfrc_bias[self.dof_ids[:9]]
        self.hR = self.data.qfrc_bias[self.dof_ids[9:18]]

        # self.mu = self.Mx @ (self.jac @ self.M_inv @ self.h + self.Jdot @ self.data.qvel[:18])

        # self.D = np.sqrt(self.Mx) @np.sqrt(self.K) + np.sqrt(self.K)@np.sqrt(self.Mx)
        # self.DL = np.sqrt(self.MxL) @ np.sqrt(self.K) + np.sqrt(self.K) @ np.sqrt(self.MxL)   
        # self.DR = np.sqrt(self.MxR) @ np.sqrt(self.K) + np.sqrt(self.K) @ np.sqrt(self.MxR)  
        
        self.DL, self.DR = compute_damping_matrices(self.MxL, self.MxR, self.K) 

        # self.DL = 2*np.sqrt(self.K)
        # self.DR = 2*np.sqrt(self.K)

        self.qL = self.data.qpos[self.dof_ids[:9]]
        self.qLdot = self.data.qvel[self.dof_ids[:9]]

        self.qR = self.data.qpos[self.dof_ids[9:18]]
        self.qRdot = self.data.qvel[self.dof_ids[9:18]]
        
        self.betaL = 2*np.sqrt(self.K_null[:9,:9])@(velocityBias[:9] - self.qLdot) + self.K_null[:9,:9] @ (PosturalBias[:9] - self.qL)  # 9 x 9 9 x 1 
        self.betaR = 2*np.sqrt(self.K_null[9:18 , 9:18])@(velocityBias[9:18] - self.qRdot) + self.K_null[9:18 , 9:18] @ (PosturalBias[9:18] - self.qR)


    def optimize(self,PosturalBias,velocityBias,JacPL,JacPR,Wimp,Wpos,Qrange,Qdotrange,tauRange):
        self.SetDynamicParams(PosturalBias,velocityBias,JacPL,JacPR)
        
        self.qLddot = cvxpy.Variable(9) # 0-8 for left arm 9-17 for right arm
        self.qRddot = cvxpy.Variable(9) # 0-8 for left arm 9-17 for right arm



        self.F1 = self.DL @ self.twistL + self.K @ self.PosErrL
        self.F2 = self.DR @ self.twistR + self.K @ self.PosErrR


        self.EimpL = self.JL @ self.qLddot + self.JLdot @ self.qLdot - self.MxL_inv @ self.F1
        self.EimpR = self.JR @ self.qRddot + self.JRdot @ self.qRdot - self.MxR_inv @ self.F2

        self.EposL = self.qLddot - self.betaL
        self.EposR = self.qRddot - self.betaR


        self.objective = cvxpy.Minimize(Wimp * cvxpy.sum_squares(self.EimpL) + Wimp * cvxpy.sum_squares(self.EimpR) 
                                        + Wpos * cvxpy.sum_squares(self.EposL) + Wpos * cvxpy.sum_squares(self.EposR))

        self.constraints = [0.5 * self.qLddot * self.dt**2 + self.qLdot * self.dt + self.qL <= Qrange[1],
                            0.5 * self.qRddot * self.dt**2 + self.qRdot * self.dt + self.qR <= Qrange[1],
                            0.5 * self.qLddot * self.dt**2 + self.qLdot * self.dt + self.qL >= Qrange[0],
                            0.5 * self.qRddot * self.dt**2 + self.qRdot * self.dt + self.qR >= Qrange[0],

                            self.qLddot*self.dt + self.qLdot <= Qdotrange[1],
                            self.qRddot*self.dt + self.qRdot <= Qdotrange[1],
                            self.qLddot*self.dt + self.qLdot >= Qdotrange[0],
                            self.qRddot*self.dt + self.qRdot >= Qdotrange[0],

                            self.ML @ self.qLddot + self.hL <= tauRange[1],
                            self.ML @ self.qLddot + self.hL >= tauRange[0],
                            self.MR @ self.qRddot + self.hR <= tauRange[1],
                            self.MR @ self.qRddot + self.hR >= tauRange[0]
                            ]
        
        self.problem = cvxpy.Problem(self.objective,self.constraints)
        
        try:
            self.loss = self.problem.solve(verbose=False)
            print("============LOSS===========: ",self.loss)
        except:
            print("----------------------infeasible-------------------------------------")
            exit(0)
        
        if(self.qLddot.value.all() != None or self.qRddot.value.all() != None):
            self.tauL = self.ML @ self.qLddot.value + self.hL
            self.data.ctrl[:9] = self.tauL
            
            self.tauR = self.MR @ self.qRddot.value + self.hR
            self.data.ctrl[9:18] = self.tauR
            
        # self.gripperCtrl("open","both")
        # return self.qLddot.value, self.qRddot.value
        return self.loss
    
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