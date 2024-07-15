import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
import matplotlib.pyplot as plt

from controllers.Impedance import Impedance

class QuinticPolynomial:
    def __init__(self, init_pos, init_vel, init_accel, final_pos, final_vel, final_accel, dist):
        # Derived coefficients
        self.a_0 = init_pos
        self.a_1 = init_vel
        self.a_2 = init_accel / 2.0

        # Solve the linear equation (Ax = B)
        A = np.array([[dist ** 3,      dist ** 4,        dist ** 5], 
                     [3 * dist ** 2,   4 * dist ** 3,    5 * dist ** 4],
                     [6 * dist,        12 * dist ** 2,   20 * dist ** 3]])

        B = np.array([final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2, 
                      final_vel - self.a_1 - 2 * self.a_2 * dist,
                      final_accel - 2 * self.a_2])

        x = np.linalg.solve(A, B)

        self.a_3 = x[0]
        self.a_4 = x[1]
        self.a_5 = x[2]

    def calc_point(self, s):
        xs = self.a_0 + self.a_1 * s + self.a_2 * s ** 2 + self.a_3 * s ** 3 + self.a_4 * s ** 4 + self.a_5 * s ** 5
        return xs

    def calc_first_derivative(self, s):
        xs = self.a_1 + 2 * self.a_2 * s + 3 * self.a_3 * s ** 2 + 4 * self.a_4 * s ** 3 + 5 * self.a_5 * s ** 4
        return xs

    def calc_second_derivative(self, s):
        xs = 2 * self.a_2 + 6 * self.a_3 * s + 12 * self.a_4 * s ** 2 + 20 * self.a_5 * s ** 3
        return xs

    def calc_third_derivative(self, s):
        xs = 6 * self.a_3 + 24 * self.a_4 * s + 60 * self.a_5 * s ** 2
        return xs

parser = ap.ArgumentParser(prog="pnp", description="simple pick and place task")

parser.add_argument("-d", "--model", type=str, help="""Choose variant of dual panda arms.
                    'dual' for individual separate arms, 'bimanual' for arms connected
                     to a torso at shoulder joint. Default is 'dual'""")

args = parser.parse_args()

if args.model == "bimanual":
    model_path = "/home/mtronlab/Shreya/OSC/MuJoCo-Dual-Arm-main/models/bimanual_panda.xml"
else:
    model_path = "/home/mtronlab/Shreya/OSC/MuJoCo-Dual-Arm-main/models/dual_panda.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False
)

Ipos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
Iori = np.asarray([30.0, 30.0, 30.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([70.0, 70.0, 35.0, 35.0, 12.5, 10.0, 2.0, 2.0, 2.0,
                      70.0, 70.0, 35.0, 35.0, 12.5, 10.0, 2.0, 2.0, 2.0])

# Damping ratio for both Cartesian and joint impedance control.
D = 1.3

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos = 4.3

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori = 4.3

# Integration timestep in seconds.
integration_dt = 0.1

# Whether to enable gravity compensation.
gravity_compensation = True

# Simulation timestep in seconds.
dt = 0.002

def create_quintic_trajectory(init_pose, final_pose, steps):
    traj = []
    x_poly = QuinticPolynomial(init_pose[0], 0, 0, final_pose[0], 0, 0, steps)
    y_poly = QuinticPolynomial(init_pose[1], 0, 0, final_pose[1], 0, 0, steps)
    z_poly = QuinticPolynomial(init_pose[2], 0, 0, final_pose[2], 0, 0, steps)
    
    for i in range(steps):
        traj.append([x_poly.calc_point(i), y_poly.calc_point(i), z_poly.calc_point(i)])
    
    return traj

def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Impedance(model, data, viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt, dt=dt,
                         gravity_compensation=gravity_compensation)
    
    controller.resetViewer()

    tolerance = 0.07
    erL = 1000
    erR = 1000

    i = 0
    j = 0

    stage = 1

    # DtrajL = create_quintic_trajectory(data.mocap_pos[controller.mocap_idL], [-0.07199763998482589, 0.001253687482995076, 0.3002852891384753], 1500)
    # DtrajR = create_quintic_trajectory(data.mocap_pos[controller.mocap_idR], [0.3693712451704328, 0.009301520145709513, -0.06959037371058782], 1500)

    # DtrajL = create_quintic_trajectory(data.mocap_pos[controller.mocap_idL], [-0.15, 0.33, 0.268], 1500)
    # DtrajR = create_quintic_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.33, 0.268], 1500)
    
    DtrajL = create_quintic_trajectory(data.mocap_pos[controller.mocap_idL], [-0.30, 0.33, 0.268], 1500)
    DtrajR = create_quintic_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.33, 0.268], 1500)  

    # Lists to store data for plotting
    time_steps = []
    erL_list = []
    erR_list = []
    posL_list = []
    posR_list = []

    jacP = controller.jac

    while viewer.is_running():
        if(erL < tolerance and erR < tolerance and i <= 1500 and stage == 1):
            data.mocap_pos[controller.mocap_idL] = DtrajL[i][:3]
            data.mocap_pos[controller.mocap_idR] = DtrajR[i][:3]
            i += 1
            if(i == 1500):
                stage += 1
                AtrajL = create_quintic_trajectory(data.mocap_pos[controller.mocap_idL], [-0.2, 0.33, 0.5], 1500)
                AtrajR = create_quintic_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.33, 0.5], 1500)

        if(erL < tolerance and erR < tolerance and j <= 1500 and stage == 2):
            data.mocap_pos[controller.mocap_idL] = AtrajL[j]
            data.mocap_pos[controller.mocap_idR] = AtrajR[j]
            j += 1
            if(j == 1500):
                stage += 1

        controller.armCrtl(jacP)

        erL = np.linalg.norm(controller.dxL)
        erR = np.linalg.norm(controller.dxR)

        # Collect data for plotting
        time_steps.append(i * dt)
        erL_list.append(erL)
        erR_list.append(erR)
        posL_list.append(data.mocap_pos[controller.mocap_idL].copy())
        posR_list.append(data.mocap_pos[controller.mocap_idR].copy())
        # forceL_list.append(controller.F_extL.copy())
        # forceR_list.append(controller.F_extR.copy())

        if(stage == 1):
            controller.gripperCtrl("open", "both")
        elif(stage == 2):
            controller.gripperCtrl("close", "both")
        elif(stage == 3):
            controller.gripperCtrl("open", "both")
            time.sleep(10)
            viewer.close()

        mujoco.mj_step(model, data)
        jacP = controller.jac
        viewer.sync()

    # Convert lists to numpy arrays for plotting
    time_steps = np.array(time_steps)
    erL_list = np.array(erL_list)
    erR_list = np.array(erR_list)
    posL_list = np.array(posL_list)
    posR_list = np.array(posR_list)

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Position trajectories
    axs[0, 0].plot(time_steps, posL_list[:, 0], label='X')
    axs[0, 0].plot(time_steps, posL_list[:, 1], label='Y')
    axs[0, 0].plot(time_steps, posL_list[:, 2], label='Z')
    axs[0, 0].set_title('Left Arm Position')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Position [m]')
    axs[0, 0].legend()

    axs[0, 1].plot(time_steps, posR_list[:, 0], label='X')
    axs[0, 1].plot(time_steps, posR_list[:, 1], label='Y')
    axs[0, 1].plot(time_steps, posR_list[:, 2], label='Z')
    axs[0, 1].set_title('Right Arm Position')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Position [m]')
    axs[0, 1].legend()

    # Error in position
    axs[1, 0].plot(time_steps, erL_list, label='Left Arm')
    axs[1, 0].plot(time_steps, erR_list, label='Right Arm')
    axs[1, 0].set_title('Position Error')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Error [m]')
    axs[1, 0].legend()

  

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
