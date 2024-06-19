import mujoco
import mujoco.msh2obj
import numpy as np

class GradientDescentIK:
    def __init__(self,model,data,step_size,tol,alpha,jacp,jacr):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr

    def checkJointLimits(self,q):
        """ check if joints are within defined joint limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
            
    def solve(self,goal,init_qpos,body_id):
        """solve for desired joint angles for goal"""
        self.data.qpos = init_qpos;
        mujoco.mj_forward(self.model,self.data)
        curr_pose = self.data.body(body_id).xpos
        error = np.subtract(goal,curr_pose)

        while(np.linalg.norm(error)>=self.tol):
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            grad = self.alpha * self.jacp.T @ error
            self.data.qpos += self.step_size * grad
            self.checkJointLimits(self.data.qpos)
            mujoco.mj_forward(self.model, self.data)
            error = np.subtract(goal, self.data.body(body_id).xpos)


class GaussNewtonIK:
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, viewer):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp1 = jacp
        self.jacp2 = jacp.copy()
        self.jacr1 = jacr
        self.jacr2 = jacr.copy()
        self.trajectory = []
        self.viewer = viewer
    
    def checkJointLimits(self, q):
        """Check if the joints are within their limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
    
    def solve(self, goal1, goal2, init_qpos, body_id1, body_id2):
        self.data.qpos[:] = init_qpos
        mujoco.mj_forward(self.model, self.data)
        current_pose1 = self.data.body(body_id1).xpos
        error1 = np.subtract(goal1, current_pose1)
        current_pose2 = self.data.body(body_id2).xpos
        error2 = np.subtract(goal2, current_pose2)
        
        while self.viewer.is_running():
            # Compute Jacobians for both arms
            mujoco.mj_jac(self.model, self.data, self.jacp1, self.jacr1, goal1, body_id1)
            mujoco.mj_jac(self.model, self.data, self.jacp2, self.jacr2, goal2, body_id2)
            
            # Compute product of Jacobians
            product1 = self.jacp1.T @ self.jacp1
            product2 = self.jacp2.T @ self.jacp2
            
            # Compute Jacobian pseudoinverses
            if np.isclose(np.linalg.det(product1), 0):
                j_inv1 = np.linalg.pinv(self.jacp1)
            else:
                j_inv1 = np.linalg.inv(product1) @ self.jacp1.T
            
            if np.isclose(np.linalg.det(product2), 0):
                j_inv2 = np.linalg.pinv(self.jacp2)
            else:
                j_inv2 = np.linalg.inv(product2) @ self.jacp2.T
            
            # Compute changes in joint positions
            delta_q1 = j_inv1 @ error1
            delta_q2 = j_inv2 @ error2

            # Update joint positions for both arms
            self.data.qpos[9:15] += self.step_size * delta_q1[9:15]
            self.data.qpos[0:6] += self.step_size * delta_q2[0:6]

            # Forward the simulation
            mujoco.mj_forward(self.model, self.data)
            
            # Check joint limits
            self.checkJointLimits(self.data.qpos)

            # Update errors
            error1 = np.subtract(goal1, self.data.body(body_id1).xpos)
            error2 = np.subtract(goal2, self.data.body(body_id2).xpos)
            
            # Store the trajectory
            self.trajectory.append(self.data.qpos.copy())

            # Perform a simulation step and sync viewer
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

            # Check if the error is within the tolerance for both arms
            if np.linalg.norm(error1) <= self.tol and np.linalg.norm(error2) <= self.tol:
                break

        


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

