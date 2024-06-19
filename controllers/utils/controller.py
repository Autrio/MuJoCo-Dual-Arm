import abc
from collections.abc import Iterable

import mujoco
import numpy as np

class controller(object, metaclass = abc.ABCMeta):
    
    def __init__(self,model,data,eef_name,actuator_range):
        self.actuator_min = actuator_range[0]
        self.actuator_max = actuator_range[1]

        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

        self.control_dim = None
        self.output_min = None
        self.output_max = None
        self.input_min = None
        self.input_max = None

        self.model = model
        self.data = data
        self.eef_name = eef_name

        self.eef_pos = None
        self.eef_ori = None
        self.eef_pos_vel = None
        self.eef_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None

        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None

        self.joint_dims = self.model.njnt

        self.torques = None
        mujoco.mj_forward(self.model,self.data)

        self.new_update = True

        self.update()
        self.initial_joint = self.joint_pos
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori = self.ee_ori

    @abc.abstractmethod
    def run_controller(self):
        self.new_update = True

    def update(self):
        if self.new_update:
            mujoco.mj_forward(self.model,self.data)
            self.ee_pos = np.array(self.data.body(self.eef_name).xpos)
            self.ee_ori= np.array(self.data.body(self.eef_name).xmat.reshape([3, 3]))

            self.ee_pos_vel = np.array(self.data.body(self.eef_name).cvel[0:2])
            self.ee_ori_vel = np.array(self.data.body(self.eef_name).cvel[3:5])

            self.joint_pos = np.array(self.data.qpos)
            self.joint_vel = np.array(self.data.qvel)

            # self.J_pos = np.array(self.data.site(self.eef_name).jacp.reshape((3, -1))[:, len(self.data.qpos)])
            # self.J_ori = np.array(self.data.site(self.eef_name).jacr.reshape((3, -1))[:, len(self.data.qvel)])

            mujoco.mj_jac(self.model, self.data,)

            self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

            mass_matrix = np.ndarray(shape=(self.model.nv, self.model.nv), dtype=np.float64, order="C")
            mujoco.mj_fullM(self.model._model, mass_matrix, self.data.qM)
            mass_matrix = np.reshape(mass_matrix, (len(self.data.qvel), len(self.data.qvel)))
            self.mass_matrix = mass_matrix[len(self.data.qvel), :][:, len(self.data.qvel)]

            self.new_update = False

    @staticmethod
    def nums2array(nums,dim):
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums
    

    def scale_action(self, action):

        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

    
    @property
    def torque_compensation(self):
        return self.sim.data.qfrc_bias[self.data.qvel]