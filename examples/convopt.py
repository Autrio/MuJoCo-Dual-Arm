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

K = np.array([500,500,500,500,500,500])
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

    
    Wimp = 10
    Wpos = 0.01


    Qrange = np.array([-176,215])
    Qdotrange = np.array([-180,180])
    tauRange = np.array([-1000,1000])

    graspIdx=56
    name = "chair"

    grasps = np.load("examples/generatedGrasps/grasp-{}.npy".format(name))
    graspL = grasps[graspIdx][1]
    graspR = grasps[graspIdx][0]
    object_scale = 1
    objStrPos = [-0.4,0.0,0.2235932541966166*object_scale]
    objStrOri = [0,0,0]

    init_pose_L = Util.eefPose(data,"end_effector")
    init_pose_R = Util.eefPose(data,"end_effector1")

    data.mocap_pos[controller.mocap_idL] = init_pose_L[:3]
    data.mocap_pos[controller.mocap_idR] = init_pose_R[:3]

    data.mocap_quat[controller.mocap_idL] = init_pose_L[3:]
    data.mocap_quat[controller.mocap_idR] = init_pose_R[3:]

    init_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
    # final_pose_L = ([-0.10, 0.33, 0.275],[0, 0, 1, 0])
    # final_pose_L = ([-0.12, 0.33, 0.4],[1, 0, 1, 0])
    final_pose_L = Util.Tmat2pose(graspL,object_scale,objStrPos,objStrOri)

    init_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))
    # final_pose_R = ([0.03, 0.33, 0.17],[0, 1, 0, -1])
    # final_pose_R = ([0.03, 0.33, 0.275], [0, 1, 0, 0])
    final_pose_R = Util.Tmat2pose(graspR,object_scale,objStrPos,objStrOri)

    # print(final_pose_L)
    # print(final_pose_R)
    # exit()

    pre_grasp_pose_L = Util.GenPreGrasp(final_pose_L,0.22)
    pre_grasp_pose_R = Util.GenPreGrasp(final_pose_R,0.22)

    
    # pre_grasp_pose_L = ([-0.10, 0.33, 0.518],[0, 0, 1, 0]) 
    # pre_grasp_pose_R = ([0.03, 0.33, 0.518], [0, 1, 0, 0])

    DtrajL_pre = create_quintic_trajectory(init_pose_L, pre_grasp_pose_L, 1500)
    DtrajR_pre = create_quintic_trajectory(init_pose_R, pre_grasp_pose_R, 1500)

    i = 0
    stage = 1
    loss = []



    while viewer.is_running():
        if( i <= 1500 and stage == 1):
            data.mocap_pos[controller.mocap_idL] = DtrajL_pre[i][:3]
            data.mocap_quat[controller.mocap_idL] = DtrajL_pre[i][3:]
            data.mocap_pos[controller.mocap_idR] = DtrajR_pre[i][:3]
            data.mocap_quat[controller.mocap_idR] = DtrajR_pre[i][3:]
            i += 1
            if(i == 1500):
                stage += 1
                logger.info("Initialising Stage Change: PRE-GRASP----->GRASP")

                current_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
                current_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))

                DtrajL = create_quintic_trajectory(current_pose_L,final_pose_L, 1500)
                DtrajR = create_quintic_trajectory(current_pose_R,final_pose_R, 1500)

        lossT = controller.optimize(postBias,velBias,jacPL,jacPR,Wimp,Wpos,Qrange,Qdotrange,tauRange)
        loss.append(lossT)

        mujoco.mj_step(model,data)

        jacPL = controller.JL
        jacPR = controller.JR

        viewer.sync()
    
    loss = np.array(loss)
    np.save("./examples/loss.npy",loss)


    # controller.makeplots()

if __name__ == "__main__":
    main()