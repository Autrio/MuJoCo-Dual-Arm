import mujoco
import numpy as np
from controllers.utils.controller import controller

class Impedance(controller):
    def __init__(self, model, data, eef_name, actuator_range, input_max=1,input_min=-1,output_max=0.05,output_min=-0.05,
                 kp=50,damping_ratio=1,impedance_mode="fixed",kp_limits=(0, 300),damping_ratio_limits=(0, 100),policy_freq=20,
                 qpos_limits=None):
        super().__init__(model, data, eef_name, actuator_range)

        self.control_dim = self.model.njnt
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        self.position_limits = np.array(qpos_limits) if qpos_limits is not None else qpos_limits

        self.kp = self.nums2array(kp, self.control_dim)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        self.kp_min = self.nums2array(kp_limits[0], self.control_dim)
        self.kp_max = self.nums2array(kp_limits[1], self.control_dim)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], self.control_dim)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], self.control_dim)

        self.impedance_mode = impedance_mode

        if self.impedance_mode == "variable":
            self.control_dim *= 3
        elif self.impedance_mode == "variable_kp":
            self.control_dim *= 2

        self.control_freq = policy_freq

        self.goal_qpos = None

    def set_goal_position(delta, current_position, position_limit=None, set_pos=None):
        n = len(current_position)
        if set_pos is not None:
            goal_position = set_pos
        else:
            goal_position = current_position + delta

        if position_limit is not None:
            if position_limit.shape != (2, n):
                raise ValueError(
                    "Position limit should be shaped (2,{}) " "but is instead: {}".format(n, position_limit.shape)
                )

            # Clip goal position
            goal_position = np.clip(goal_position, position_limit[0], position_limit[1])

        return goal_position

    def set_goal(self,action,set_qpos=None):
        self.update()

        jnt_dim = len(self.model.qpos)
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = action[:jnt_dim], action[jnt_dim : 2 * jnt_dim], action[2 * jnt_dim :]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:jnt_dim], action[jnt_dim:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        else:  # This is case "fixed"
            delta = action

        assert len(delta) == jnt_dim, "Delta qpos must be equal to the robot's joint dimension space!"

        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

        self.goal_qpos = self.set_goal_position(
            scaled_delta, self.joint_pos, position_limit=self.position_limits, set_pos=set_qpos
        )

    def run_controller(self):

        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        self.update()

        desired_qpos = None

        desired_qpos = np.array(self.goal_qpos)

        position_error = desired_qpos - self.joint_pos
        vel_pos_error = -self.joint_vel
        desired_torque = np.multiply(np.array(position_error), np.array(self.kp)) + np.multiply(vel_pos_error, self.kd)

        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation

        super().run_controller()

        return self.torques