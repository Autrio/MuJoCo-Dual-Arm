# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# import argparse as ap
# import matplotlib.pyplot as plt

# from controllers.Impedance import Impedance
# from examples.s_rrt import generate_smooth_rrt  # Import the smooth RRT generator


# parser = ap.ArgumentParser(prog="pnp", description="simple pick and place task")

# parser.add_argument("-d", "--model", type=str, help="""Choose variant of dual panda arms.
#                     'dual' for individual separate arms, 'bimanual' for arms connected
#                      to a torso at shoulder joint. Default is 'dual'""")

# args = parser.parse_args()

# if args.model == "bimanual":
#     model_path = "/home/mtronlab/Shreya/OSC/MuJoCo-Dual-Arm-main/models/bimanual_panda.xml"
# else:
#     model_path = "/home/mtronlab/Shreya/OSC/MuJoCo-Dual-Arm-main/models/dual_panda.xml"

# model = mujoco.MjModel.from_xml_path(model_path)
# data = mujoco.MjData(model)
# viewer = mujoco.viewer.launch_passive(
#     model=model,
#     data=data,
#     show_left_ui=False,
#     show_right_ui=False
# )

# Ipos = np.asarray([80.0, 80.0, 80.0])  # [N/m]
# Iori = np.asarray([20.0, 20.0, 20.0])  # [Nm/rad]

# Kp_null = np.asarray([50.0, 50.0, 20.0, 20.0, 5, 5.0, 1.0, 1.0, 1.0,
#                       50.0, 50.0, 20.0, 20.0, 5, 5.0, 1.0, 1.0, 1.0])

# D = 1.3

# Kpos = 4.3
# Kori = 4.3

# integration_dt = 0.1
# gravity_compensation = True
# dt = 0.002

# def create_rrt_trajectory(init_pose, final_pose, steps):
#     traj = generate_smooth_rrt(init_pose, final_pose, steps)
#     return traj

# def main():
#     assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

#     controller = Impedance(model, data, viewer)

#     controller.setParams(Ipos=Ipos, Iori=Iori,
#                          Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
#                          D=D, integration_dt=integration_dt, dt=dt,
#                          gravity_compensation=gravity_compensation)
    
#     controller.resetViewer()

#     tolerance = 0.05
#     erL = 1000
#     erR = 1000

#     i = 0
#     j = 0

#     stage = 1

#     DtrajL = create_rrt_trajectory(data.mocap_pos[controller.mocap_idL], [-0.15, 0.33, 0.268], 1500)
#     DtrajR = create_rrt_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.33, 0.268], 1500)

#     time_steps = []
#     erL_list = []
#     erR_list = []
#     posL_list = []
#     posR_list = []

#     jacP = controller.jac

#     while viewer.is_running():
#         if(erL < tolerance and erR < tolerance and i <= 1500 and stage == 1):
#             data.mocap_pos[controller.mocap_idL] = DtrajL[i]
#             data.mocap_pos[controller.mocap_idR] = DtrajR[i]
#             i += 1
#             if(i == 1500):
#                 stage += 1
#                 AtrajL = create_rrt_trajectory(data.mocap_pos[controller.mocap_idL], [-0.2, 0.33, 0.5], 1500)
#                 AtrajR = create_rrt_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.33, 0.5], 1500)

#         if(erL < tolerance and erR < tolerance and j <= 1500 and stage == 2):
#             data.mocap_pos[controller.mocap_idL] = AtrajL[j]
#             data.mocap_pos[controller.mocap_idR] = AtrajR[j]
#             j += 1
#             if(j == 1500):
#                 stage += 1

#         controller.armCrtl(jacP)

#         erL = np.linalg.norm(controller.dxL)
#         erR = np.linalg.norm(controller.dxR)

#         time_steps.append(i * dt)
#         erL_list.append(erL)
#         erR_list.append(erR)
#         posL_list.append(data.mocap_pos[controller.mocap_idL].copy())
#         posR_list.append(data.mocap_pos[controller.mocap_idR].copy())

#         if(stage == 1):
#             controller.gripperCtrl("open", "both")
#         elif(stage == 2):
#             controller.gripperCtrl("close", "both")
#         elif(stage == 3):
#             controller.gripperCtrl("open", "both")
#             time.sleep(10)
#             viewer.close()

#         mujoco.mj_step(model, data)
#         jacP = controller.jac
#         viewer.sync()

#     time_steps = np.array(time_steps)
#     erL_list = np.array(erL_list)
#     erR_list = np.array(erR_list)
#     posL_list = np.array(posL_list)
#     posR_list = np.array(posR_list)

#     fig, axs = plt.subplots(2, 2, figsize=(15, 10))

#     axs[0, 0].plot(time_steps, posL_list[:, 0], label='X')
#     axs[0, 0].plot(time_steps, posL_list[:, 1], label='Y')
#     axs[0, 0].plot(time_steps, posL_list[:, 2], label='Z')
#     axs[0, 0].set_title('Left Arm Position')
#     axs[0, 0].set_xlabel('Time [s]')
#     axs[0, 0].set_ylabel('Position [m]')
#     axs[0, 0].legend()

#     axs[0, 1].plot(time_steps, posR_list[:, 0], label='X')
#     axs[0, 1].plot(time_steps, posR_list[:, 1], label='Y')
#     axs[0, 1].plot(time_steps, posR_list[:, 2], label='Z')
#     axs[0, 1].set_title('Right Arm Position')
#     axs[0, 1].set_xlabel('Time [s]')
#     axs[0, 1].set_ylabel('Position [m]')
#     axs[0, 1].legend()

#     axs[1, 0].plot(time_steps, erL_list, label='Left Arm')
#     axs[1, 0].plot(time_steps, erR_list, label='Right Arm')
#     axs[1, 0].set_title('Position Error')
#     axs[1, 0].set_xlabel('Time [s]')
#     axs[1, 0].set_ylabel('Error [m]')
#     axs[1, 0].legend()

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()


import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
import matplotlib.pyplot as plt

from controllers.Impedance import Impedance
from examples.s_rrt import generate_smooth_rrt


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

def create_rrt_trajectory(start, goal, steps, obstacle_list):
    path = generate_smooth_rrt(start, goal, steps, obstacle_list)
    return path

def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Impedance(model, data, viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt, dt=dt,
                         gravity_compensation=gravity_compensation)
    
    controller.resetViewer()

    tolerance = 0.05
    erL = 1000
    erR = 1000

    i = 0
    j = 0

    stage = 1

    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]

    DtrajL = create_rrt_trajectory(data.mocap_pos[controller.mocap_idL], [-0.25, 0.33, 0.15], 1500, obstacle_list)
    DtrajR = create_rrt_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.33, 0.268], 1500, obstacle_list)

    time_steps = []
    erL_list = []
    erR_list = []
    posL_list = []
    posR_list = []

    jacP = controller.jac

    while viewer.is_running():
        if(erL < tolerance and erR < tolerance and i <= 1500 and stage == 1):
            data.mocap_pos[controller.mocap_idL] = DtrajL[i]
            data.mocap_pos[controller.mocap_idR] = DtrajR[i]
            i += 1
            if(i == 1500):
                print("Stage 1 completed")
                stage += 1
                obstacle_list1 = [
                    (250, 250, 250),
                    (250, 250, 250),
                    (250, 250, 250),
                    (250, 250, 250),
                    (250, 250, 250),
                    (250, 250, 250)
                ]
                AtrajL = create_rrt_trajectory(data.mocap_pos[controller.mocap_idL], [-0.15, 0.35, 0.5], 1500, obstacle_list1)
                AtrajR = create_rrt_trajectory(data.mocap_pos[controller.mocap_idR], [0.05, 0.35, 0.5], 1500, obstacle_list1)
            

        if(erL < tolerance and erR < tolerance and j <= 1500 and stage == 2):
            print("Stage 2")
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
    # np.savez('/home/mtronlab/Shreya/OSC/MuJoCo-Dual-Arm-main/logs/pend_traj_logs', erL=erL_list, erR=erR_list,
    #          posL=posL_list, posR=posR_list, ts=time_steps)

    # Convert lists to numpy arrays for plotting
    time_steps = np.array(time_steps)
    erL_list = np.array(erL_list)
    erR_list = np.array(erR_list)
    posL_list = np.array(posL_list)
    posR_list = np.array(posR_list)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

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
