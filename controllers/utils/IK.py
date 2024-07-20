import mujoco
import mujoco.msh2obj
import numpy as np
from .utils import RotationUtils

class GradientDescentIK:
    def __init__(self,model,data,step_size,tol,alpha,jacp,jacr):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.site_quat = np.zeros(4)

    def checkJointLimits(self,q):
        """ check if joints are within defined joint limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
            
    def solve(self,goal,init_qpos,site_id):
        """solve for desired joint angles for goal"""
        self.data.qpos = init_qpos[:18];
        mujoco.mj_forward(self.model,self.data)
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        curr_pos = self.data.site(site_id).xpos
        curr_pose = np.concatenate(curr_pos,self.site_quat[:3])
        error = np.subtract(goal,curr_pose)

        while(np.linalg.norm(error)>=self.tol):
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, goal, site_id)
            grad = self.alpha * self.jacp.T @ error
            self.data.qpos += self.step_size * grad
            self.checkJointLimits(self.data.qpos)
            mujoco.mj_forward(self.model, self.data)
            mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
            curr_pos = self.data.site(site_id).xpos
            curr_pose = np.concatenate(curr_pos,self.site_quat[:3])
            error = np.subtract(goal,curr_pose)



class GaussNewtonIK:
    
    def __init__(self, model, data, step_size, tol, alpha,jac1,jac2, viewer):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.trajectory = []
        self.viewer = viewer
        self.site_quat1 = np.zeros(4)
        self.site_quat2 = np.zeros(4)
        self.jac1 = jac1
        self.jac2 = jac2

    def checkJointLimits(self, q):
        """Check if the joints are within their limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    def solve(self, goal1, goal2, init_qpos, site_id1, site_id2):
        Rfunc = RotationUtils()
        self.data.qpos[:18] = init_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Process first goal
        mujoco.mju_mat2Quat(self.site_quat1, self.data.site(site_id1).xmat)
        curr_pos1 = self.data.site(site_id1).xpos
        curr_rot1 = Rfunc.Quat2rot(self.site_quat1,"wxyz","xyz",False)
        curr_pose1 = np.concatenate((curr_pos1, curr_rot1))
        error1 = np.subtract(goal1, curr_pose1)
        
        # Process second goal
        mujoco.mju_mat2Quat(self.site_quat2, self.data.site(site_id2).xmat)
        curr_pos2 = self.data.site(site_id2).xpos
        curr_rot2 = Rfunc.Quat2rot(self.site_quat2,"wxyz","xyz",False)
        curr_pose2 = np.concatenate((curr_pos2, curr_rot2))
        error2 = np.subtract(goal2, curr_pose2)

        while self.viewer.is_running():
            # Compute Jacobians for both arms
            mujoco.mj_jacSite(self.model, self.data, self.jac1[:3], self.jac1[3:], site_id1)
            mujoco.mj_jacSite(self.model, self.data, self.jac2[:3], self.jac2[3:], site_id2)

            self.Jac1 = self.jac1[:, :9]
            self.Jac2 = self.jac2[:, 9:18]

            # Compute product of Jacobians
            product1 = self.Jac1.T @ self.Jac1
            product2 = self.Jac2.T @ self.Jac2

            # Compute Jacobian pseudoinverses
            if np.isclose(np.linalg.det(product1), 0):
                j_inv1 = np.linalg.pinv(self.Jac1)
            else:
                j_inv1 = np.linalg.inv(product1) @ self.Jac1.T

            if np.isclose(np.linalg.det(product2), 0):
                j_inv2 = np.linalg.pinv(self.Jac2)
            else:
                j_inv2 = np.linalg.inv(product2) @ self.Jac2.T

            # Compute changes in joint positions
            delta_q1 = j_inv1 @ error1
            delta_q2 = j_inv2 @ error2

            # Update joint positions for both arms
            self.data.qpos[0:9] += self.step_size * delta_q1
            self.data.qpos[9:18] += self.step_size * delta_q2

            # Forward the simulation
            mujoco.mj_forward(self.model, self.data)
            
            # Check joint limits
            self.checkJointLimits(self.data.qpos[:18])

            # Update errors
            mujoco.mju_mat2Quat(self.site_quat1, self.data.site(site_id1).xmat)
            curr_pos1 = self.data.site(site_id1).xpos
            curr_rot1 = Rfunc.Quat2rot(self.site_quat1,"wxyz","xyz",False)
            curr_pose1 = np.concatenate((curr_pos1, curr_rot1))
            error1 = np.subtract(goal1, curr_pose1)
            
            # Process second goal
            mujoco.mju_mat2Quat(self.site_quat2, self.data.site(site_id2).xmat)
            curr_pos2 = self.data.site(site_id2).xpos
            curr_rot2 = Rfunc.Quat2rot(self.site_quat2,"wxyz","xyz",False)
            curr_pose2 = np.concatenate((curr_pos2, curr_rot2))
            error2 = np.subtract(goal2, curr_pose2)

            
            # Store the trajectory
            self.trajectory.append(self.data.qpos.copy())

            # Perform a simulation step and sync viewer
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            # Check if the error is within the tolerance for both arms
            if np.linalg.norm(error1) <= self.tol and np.linalg.norm(error2) <= self.tol:
                return self.trajectory

            

        


class LevenbergMarquardtIK:
    def __init__(self,model,data,step_size,tol,alpha,jacp,jacr,damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping  = damping

    def checkJointLimits(self,q):
        """ check if joints are within defined joint limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
            
    def solve(self, goal, init_qpos, body_id):
        self.data.qpos = init_qpos
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)

        while(np.linalg.norm(error)>=self.tol):
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            n = self.jacp.shape[1]
            I = np.identity
            prod = self.jacp.T @ self.jacp + self.damping * I

            if np.isclose(np.linalg.det(prod),0):
                j_inv = np.linalg.pinv(prod) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(prod) @ self.jacp.T

            del_q = j_inv @ error

            self.data.qpos += self.step_size * del_q
            self.checkJointLimits(self.data.qpos)
            mujoco.mj_forward(self.model, self.data)
            error = np.subtract(goal, self.data.body(body_id).xpos)

