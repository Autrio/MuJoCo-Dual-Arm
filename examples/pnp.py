import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from controllers.Impedance import Impedance
from controllers.utils.QuinticPolynomial import *
from controllers.utils.utils import *


parser = ap.ArgumentParser(prog="pnp", description="simple pick and place task")

parser.add_argument("-d", "--model", type=str, help="""Choose variant of dual panda arms.
                    'dual' for individual separate arms, 'bimanual' for arms connected
                     to a torso at shoulder joint. Default is 'dual'""")

args = parser.parse_args()

if(args.model == "bimanual"):
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/bimanual_panda.xml";
else:
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False
)

Ipos = np.asarray([500.0, 500.0, 500.0])  # [N/m]
Iori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]
Kp_null = np.asarray([70.0, 70.0, 35.0, 35.0, 12.5, 10.0, 2.0, 2.0, 2.0,
                      70.0, 70.0, 35.0, 35.0, 12.5, 10.0, 2.0, 2.0, 2.0])
D = 1.3
Kpos = 6
Kori = 4
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

    tolerance = 0.04
    erL = 1000
    erR = 1000

    i = 0
    j = 0
    k = 0

    stage = 1
    
    Util = RotationUtils()

    init_pose_L = Util.eefPose(data,"end_effector")
    init_pose_R = Util.eefPose(data,"end_effector1")

    data.mocap_pos[controller.mocap_idL] = init_pose_L[:3]
    data.mocap_pos[controller.mocap_idR] = init_pose_R[:3]

    data.mocap_quat[controller.mocap_idL] = init_pose_L[3:]
    data.mocap_quat[controller.mocap_idR] = init_pose_R[3:]


    init_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
    final_pose_L = ([-0.10, 0.33, 0.275],[0, 0, 1, 0])
    # final_pose_L = ([-0.12, 0.33, 0.4],[1, 0, 1, 0])
    init_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))
    # final_pose_R = ([0.03, 0.33, 0.17],[0, 1, 0, -1])
    final_pose_R = ([0.03, 0.33, 0.275], [0, 1, 0, 0])
    
    pre_grasp_pose_L = ([-0.10, 0.33, 0.518],[0, 0, 1, 0]) 
    pre_grasp_pose_R = ([0.03, 0.33, 0.518], [0, 1, 0, 0])

    DtrajL_pre = create_quintic_trajectory(init_pose_L, pre_grasp_pose_L, 1500)
    DtrajR_pre = create_quintic_trajectory(init_pose_R, pre_grasp_pose_R, 1500)

    time_steps = []
    erL_list = []
    erR_list = []
    posL_list = []
    posR_list = []
    erL_joint = []
    erR_joint = []
    forceL_list = []
    forceR_list = []

    jacP = controller.jac


    while viewer.is_running():
        if(erL < tolerance and erR < tolerance and i <= 1500 and stage == 1):
            data.mocap_pos[controller.mocap_idL] = DtrajL_pre[i][:3]
            data.mocap_quat[controller.mocap_idL] = DtrajL_pre[i][3:]
            data.mocap_pos[controller.mocap_idR] = DtrajR_pre[i][:3]
            data.mocap_quat[controller.mocap_idR] = DtrajR_pre[i][3:]
            i += 1
            if(i == 1500):
                stage += 1
                current_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
                current_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))

                DtrajL = create_quintic_trajectory(current_pose_L,final_pose_L, 1500)
                DtrajR = create_quintic_trajectory(current_pose_R,final_pose_R, 1500)
            
        if(erL < tolerance and erR < tolerance and j <= 1500 and stage == 2):
            data.mocap_pos[controller.mocap_idL] = DtrajL[j][:3]
            data.mocap_quat[controller.mocap_idL] = DtrajL[j][3:]
            data.mocap_pos[controller.mocap_idR] = DtrajR[j][:3]
            data.mocap_quat[controller.mocap_idR] = DtrajR[j][3:]
            j += 1
            if(j == 1500):
                stage += 1
                current_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
                current_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))

                init_object_pose = ([0.0, 0.3,0.0], [0, 0, 0, 1])
                final_object_pose = ([0, 0.3,0.3], [1,0, 0, 1])

                # Create a single trajectory for the object's center of mass
                object_trajectory = create_quintic_trajectory(init_object_pose, final_object_pose, 1500)
                AtrajL, AtrajR = generate_end_effector_trajectories(object_trajectory, init_object_pose, current_pose_L, current_pose_R)

        
        if(erL < tolerance and erR < tolerance and k <= 1500 and stage == 3):
            data.mocap_pos[controller.mocap_idL] = AtrajL[k][:3]
            data.mocap_quat[controller.mocap_idL] = AtrajL[k][3:]
            data.mocap_pos[controller.mocap_idR] = AtrajR[k][:3]
            data.mocap_quat[controller.mocap_idR] = AtrajR[k][3:]

            k += 1
            if(k == 1500):
                stage += 1


 
        controller.armCtrl(jacP)

        erL = np.linalg.norm(controller.dxL)
        erR = np.linalg.norm(controller.dxR)
        # print("Error Left: ", erL_joint)

        # Collect data for plotting
        time_steps.append(len(time_steps) * dt)
        erL_list.append(erL)
        erR_list.append(erR)
        posL_list.append(data.mocap_pos[controller.mocap_idL].copy())
        posR_list.append(data.mocap_pos[controller.mocap_idR].copy())
        erL_joint.append(np.linalg.norm(controller.joint_errorL))  # Example of joint angle error collection
        erR_joint.append(np.linalg.norm(controller.joint_errorR))  # Example of joint angle error collection

        forceL_list.append(np.linalg.norm(controller.tau[8]))  # Example of force collection
        forceR_list.append(np.linalg.norm(controller.tau[17]))  # Example of force collection


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

    # Convert lists to numpy arrays for easier manipulation
    time_steps = np.array(time_steps)
    erL_list = np.array(erL_list)
    erR_list = np.array(erR_list)
    posL_list = np.array(posL_list)
    posR_list = np.array(posR_list)
    erL_joint = np.array(erL_joint)
    erR_joint = np.array(erR_joint)
    forceL_list = np.array(forceL_list)
    forceR_list = np.array(forceR_list)


    # Plot position errors in Cartesian space for both arms
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 2, 1)
    plt.plot(time_steps, erL_list, label='Left Arm Position Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Left Arm Position Error in Cartesian Space')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_steps, erR_list, label='Right Arm Position Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Right Arm Position Error in Cartesian Space')
    plt.legend()

    # Plot joint angle errors for both arms
    plt.subplot(3, 2, 3)
    plt.plot(time_steps, erL_joint, label='Left Arm Joint Angle Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle Error (rad)')
    plt.title('Left Arm Joint Angle Error')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time_steps, erR_joint, label='Right Arm Joint Angle Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle Error (rad)')
    plt.title('Right Arm Joint Angle Error')
    plt.legend()

    # plt.tight_layout()
    # plt.show()

    # # Plot force of the end effector for each timestep
    # plt.figure(figsize=(12, 6))

    plt.subplot(3, 2, 5)
    plt.plot(time_steps, forceL_list, label='Left Arm End Effector Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Left Arm End Effector Force')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time_steps, forceR_list, label='Right Arm End Effector Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Right Arm End Effector Force')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()