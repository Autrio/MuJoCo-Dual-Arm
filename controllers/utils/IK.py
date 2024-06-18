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
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr,viewer):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.trajectory = []
        self.viewer = viewer
    
    def checkJointLimits(self, q):
        """Check if the joints are within their limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
    
    def solve(self, goal, init_qpos, body_id):
        self.data.qpos = init_qpos
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)
        
        
        while (self.viewer.is_running() or np.linalg.norm(error) >= self.tol):
            mujoco.mj_jac(self.model, self.data, self.jacp, 
                          self.jacr, goal, body_id)
            product = self.jacp.T @ self.jacp
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            self.data.qpos += self.step_size * delta_q
            self.checkJointLimits(self.data.qpos)
            mujoco.mj_forward(self.model, self.data) 
            error = np.subtract(goal, self.data.body(body_id).xpos) 
            self.trajectory.append(self.data.xpos)

            mujoco.mj_step(self.model,self.data)
            self.viewer.sync()


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

