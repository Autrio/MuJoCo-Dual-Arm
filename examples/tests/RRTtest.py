import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap

from controllers.Impedance import Impedance
from controllers.utils.RRT import RRT,StateSpace,GoalSpace

parser = ap.ArgumentParser(prog="pnp",
                           description="simple pick and place task")

parser.add_argument("-d", "--model", type=str, help="""Choose variant of dual panda arms.
                    'dual' for individual separate arms, 'bimanual' for arms connected
                     to a torso at shoulder joint. Default is 'dual'""")

args = parser.parse_args()


if (args.model == "bimanual"):
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/bimanual_panda.xml"
else:
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False)

Ipos = np.asarray([500.0, 500.0, 500.0])  # [N/m]
Iori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([70.0, 70.0, 35.0, 35.0, 12.5, 10.0, 2.0, 2.0, 2.0,
                      70.0, 70.0, 35.0, 35.0, 12.5, 10.0, 2.0, 2.0, 2.0])

# Damping ratio for both Cartesian and joint impedance control.
D = 1.3
# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 4

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 4

# Integration timestep in seconds.
integration_dt: float = 0.1

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002


def plan(goal_qpos, start_qpos, rrt_range=0.1, rrt_max_iter=10000, start_qpos_range=0.0,
         start_qpos_max_trials=100, seed=None):


    stateSpace = StateSpace(model.jnt_range.min(),model.jnt_range.max())
    stateSpace.InitModel(model,data)
    goalSpace = GoalSpace(goal_qpos,stateSpace,0.01)

    rrt = RRT(stateSpace)
    rrt.setParams([start_qpos],goalSpace.goal,maxDist=rrt_range,maxIter=rrt_max_iter,startStateRange=start_qpos_range,
                  startStateMaxTrials=start_qpos_max_trials,seed=seed)
    
    path = rrt.solve()

    return path

    


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    model.opt.gravity = 0.0

    controller = Impedance(model, data, viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt, dt=dt,
                         gravity_compensation=gravity_compensation)

    controller.resetViewer()

    goal_qpos = [0.57329266, -0.8592102,   1.13909264, -2.63463105,  1.84819556,  1.09160145, -0.25412081,  0.04001436,  0.04117879 ]
    start_qpos = data.qpos[:9]
    print(start_qpos)
    exit

    path = plan(goal_qpos,start_qpos,0.1,10000,0.0,100,100)
    print(path)
    exit()
    # i = 0
    # while(viewer.is_running()):
    #     if(i<len(path)):
    #         data.qpos[:9]=path[i]
    #         mujoco.mj_forward(model,data)
    #         i+=1
    #     mujoco.mj_step(model,data)
    #     viewer.sync()

if __name__ == "__main__":
    main()