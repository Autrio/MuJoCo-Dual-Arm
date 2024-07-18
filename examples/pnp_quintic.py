import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from controllers.Impedance import Impedance



def pre_grasp_pose(grasp_pose, r, grasp_axis_local):
    # Extract position and orientation
    ([x, y, z], [qx, qy, qz, qw]) = grasp_pose

    # Convert quaternion to rotation matrix
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()

    # Convert the grasp axis from local to world frame
    grasp_axis_world = rotation_matrix @ grasp_axis_local

    # Compute the offset position
    offset_position = np.array([x, y, z]) - r * grasp_axis_world

    # Create the pre-grasp pose
    pre_grasp_pose = ([
        offset_position[0],
        offset_position[1],
        offset_position[2]],
        [qx, qy, qz, qw])

    return pre_grasp_pose

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

def create_quintic_trajectory(init_pose, final_pose, steps):
    traj = []

    # Position interpolation
    x_poly = QuinticPolynomial(init_pose[0][0], 0, 0, final_pose[0][0], 0, 0, steps)
    y_poly = QuinticPolynomial(init_pose[0][1], 0, 0, final_pose[0][1], 0, 0, steps)
    z_poly = QuinticPolynomial(init_pose[0][2], 0, 0, final_pose[0][2], 0, 0, steps)

    # Quaternion interpolation
    qx_poly = QuinticPolynomial(init_pose[1][0], 0, 0, final_pose[1][0], 0, 0, steps)
    qy_poly = QuinticPolynomial(init_pose[1][1], 0, 0, final_pose[1][1], 0, 0, steps)
    qz_poly = QuinticPolynomial(init_pose[1][2], 0, 0, final_pose[1][2], 0, 0, steps)
    qw_poly = QuinticPolynomial(init_pose[1][3], 0, 0, final_pose[1][3], 0, 0, steps)

    for i in range(steps):
        pos = [x_poly.calc_point(i), y_poly.calc_point(i), z_poly.calc_point(i)]
        quat = [qx_poly.calc_point(i), qy_poly.calc_point(i), qz_poly.calc_point(i), qw_poly.calc_point(i)]
        quat /= np.linalg.norm(quat)  # Normalize quaternion to ensure it remains valid
        traj.append(pos + quat.tolist())

    return traj

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

Ipos = np.asarray([80.0, 80.0, 80.0])  # [N/m]
Iori = np.asarray([20.0, 20.0, 20.0])  # [Nm/rad]

Kp_null = np.asarray([50.0, 50.0, 20.0, 20.0, 5, 5.0, 1.0, 1.0, 1.0,
                      50.0, 50.0, 20.0, 20.0, 5, 5.0, 1.0, 1.0, 1.0])
D = 1.3
Kpos = 4.3
Kori = 4.3
integration_dt = 0.1
gravity_compensation = True
dt = 0.002

def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Impedance(model, data, viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt, dt=dt,
                         gravity_compensation=gravity_compensation)
    
    controller.resetViewer()

    tolerance = 0.1
    erL = 1000
    erR = 1000

    i = 0
    j = 0
    k = 0

    stage = 1


    init_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
    final_pose_L = ([-0.15, 0.33, 0.23],[0, 0, 1, 0])
    # final_pose_L = ([-0.12, 0.33, 0.4],[1, 0, 1, 0])
    init_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))
    # final_pose_R = ([0.03, 0.33, 0.17],[0, 1, 0, -1])
    final_pose_R = ([0.05, 0.33, 0.23], [0, 1, 0, 0])
    

    # pre_grasp_pose_L = pre_grasp_pose(final_pose_L, 0.25, [-1, 0, 0])
    # pre_grasp_pose_R = pre_grasp_pose(final_pose_R, 0.25, [0,0 , 1]) 

    pre_grasp_pose_L = ([-0.15, 0.33, 0.518],[0, 0, 1, 0]) 
    pre_grasp_pose_R = ([0.05, 0.33, 0.518], [0, 1, 0, 0])
    # pre_grasp_pose_L = ([-0.45, 0.33, 0.07],[1, 0, 1, 0]) 
    # pre_grasp_pose_R = ([0.3, 0.33, 0.17], [0, 1, 0, -1]) 

    print("Intial Pose Left: ", init_pose_L)
    print("Final Pose Left: ", final_pose_L)
    print("Intial Pose Right: ", init_pose_R)
    print("Final Pose Right: ", final_pose_R)      

    DtrajL_pre = create_quintic_trajectory(init_pose_L, pre_grasp_pose_L, 1500)
    DtrajR_pre = create_quintic_trajectory(init_pose_R, pre_grasp_pose_R, 1500)

    time_steps = []
    erL_list = []
    erR_list = []
    posL_list = []
    posR_list = []

    jacP = controller.jac


    while viewer.is_running():
        if(erL < tolerance and erR < tolerance and i <= 1500 and stage == 1):
            print("Stage 1")
            data.mocap_pos[controller.mocap_idL] = DtrajL_pre[i][:3]
            data.mocap_quat[controller.mocap_idL] = DtrajL_pre[i][3:]
            data.mocap_pos[controller.mocap_idR] = DtrajR_pre[i][:3]
            data.mocap_quat[controller.mocap_idR] = DtrajR_pre[i][3:]
            i += 1
            if(i == 1500):
                print("Stage 2")
                stage += 1
                current_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
                current_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))

                DtrajL = create_quintic_trajectory(current_pose_L,final_pose_L, 1500)
                DtrajR = create_quintic_trajectory(current_pose_R,final_pose_R, 1500)
            
        if(erL < tolerance and erR < tolerance and j <= 1500 and stage == 2):
            print("Stage 3")
            data.mocap_pos[controller.mocap_idL] = DtrajL[j][:3]
            data.mocap_quat[controller.mocap_idL] = DtrajL[j][3:]
            data.mocap_pos[controller.mocap_idR] = DtrajR[j][:3]
            data.mocap_quat[controller.mocap_idR] = DtrajR[j][3:]
            j += 1
            if(j == 1500):
                stage += 1
                # AtrajL = create_quintic_trajectory((data.mocap_pos[controller.mocap_idL], data.mocap_quat[controller.mocap_idL]), ([-0.12, 0.33, 0.4], [1, 0, 1, 0]), 1500)
                # AtrajR = create_quintic_trajectory((data.mocap_pos[controller.mocap_idR], data.mocap_quat[controller.mocap_idR]), ([0.03, 0.33, 0.5], [0, 1, 0, -1]), 1500)

                # AtrajL = create_quintic_trajectory((data.mocap_pos[controller.mocap_idL], data.mocap_quat[controller.mocap_idL]), ([-0.15, 0.33, 0.5],[0, 0, 1, 0]), 1500)
                # AtrajR = create_quintic_trajectory((data.mocap_pos[controller.mocap_idR], data.mocap_quat[controller.mocap_idR]), ([0.05, 0.33, 0.5], [0, 1, 0, 0]), 1500)

                AtrajL = create_quintic_trajectory((data.mocap_pos[controller.mocap_idL], data.mocap_quat[controller.mocap_idL]), ([-0.25, 0.45, 0.5],[0, 0, 1, 0]), 1500)
                AtrajR = create_quintic_trajectory((data.mocap_pos[controller.mocap_idR], data.mocap_quat[controller.mocap_idR]), ([-0.05, 0.45, 0.5], [0, 1, 0, 0]), 1500)

        
        if(erL < tolerance and erR < tolerance and k <= 1500 and stage == 3):
            print("Stage 4")
            data.mocap_pos[controller.mocap_idL] = AtrajL[k][:3]
            data.mocap_quat[controller.mocap_idL] = AtrajL[k][3:]
            data.mocap_pos[controller.mocap_idR] = AtrajR[k][:3]
            data.mocap_quat[controller.mocap_idR] = AtrajR[k][3:]
            k += 1
            if(k == 1500):
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
            controller.gripperCtrl("open", "both")
        elif(stage == 3):
            controller.gripperCtrl("close", "both")
        elif(stage == 4):
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
    plt.plot(time_steps, erL_list, label='Left Limb Error')
    plt.plot(time_steps, erR_list, label='Right Limb Error')
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Error (m)")
    plt.show()

    plt.plot(time_steps, posL_list, label='Left Limb Position')
    plt.plot(time_steps, posR_list, label='Right Limb Position')
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Position (m)")
    plt.show()

 

if __name__ == "__main__":
    main()