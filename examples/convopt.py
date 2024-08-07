import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
from scipy.spatial.transform import Rotation as R
import logging 


from controllers.convex import Convex
from controllers.utils.QuinticPolynomial import *
from controllers.utils.utils import *

model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger("CONTROLLER")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False)

K = np.array([100,100,100,100,100,100])
K_null = np.array([100.0, 100.0, 55.0, 55.0, 22.5, 20.0, 5.0, 2.0, 2.0,
                      100.0, 100.0, 55.0, 55.0, 22.5, 20.0, 5.0, 2.0, 2.0])

KvPos = 1
KvOri = 1

dt = 0.002
model.opt.timestep = dt


def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Convex(model, data, viewer)

    Util = RotationUtils()

    controller.SetStaticParams(K,K_null,KvPos,KvOri,dt)

    controller.resetViewer(True)

    postBias = data.qpos[controller.dof_ids[:18]]
    velBias = data.qvel[controller.dof_ids[:18]]

    jacPL = controller.JL
    jacPR = controller.JR

    
    Wimp = 3
    Wpos = 1

    Qrange = np.array([-176,215])
    Qdotrange = np.array([0,180])
    tauRange = np.array([-100,100])

    init_pose_L = Util.eefPose(data,"end_effector")
    init_pose_R = Util.eefPose(data,"end_effector1")

    data.mocap_pos[controller.mocap_idL] = init_pose_L[:3]
    data.mocap_pos[controller.mocap_idR] = init_pose_R[:3]

    data.mocap_quat[controller.mocap_idL] = init_pose_L[3:]
    data.mocap_quat[controller.mocap_idR] = init_pose_R[3:]


    while viewer.is_running():

        controller.optimize(postBias,velBias,jacPL,jacPR,Wimp,Wpos,Qrange,Qdotrange,tauRange)

        mujoco.mj_step(model,data)

        jacPL = controller.JL
        jacPR = controller.JR

        viewer.sync()
    


    # controller.makeplots()

if __name__ == "__main__":
    main()