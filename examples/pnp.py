import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
from scipy.spatial.transform import Rotation as R
import logging 


from controllers.Impedance import Impedance
from controllers.utils.QuinticPolynomial import *
from controllers.utils.utils import *


parser = ap.ArgumentParser(prog="pnp", description="simple pick and place task")

parser.add_argument("-d", "--model", type=str, help="""Choose variant of dual panda arms.
                    'dual' for individual separate arms, 'bimanual' for arms connected
                     to a torso at shoulder joint. Default is 'dual'""")

parser.add_argument("-m","--toggle-mocap", type=str, help=""" Choose whether to toggle MoCap visuals
                    'True' for visuals, 'False' otherwise default is True""")

parser.add_argument("-t","--tolerance",type=float,help="""Set trajectory following error limit
                    default 0.04 units
                    NOTE: Tendency to stall if tolerance is too low (lower than steady state error)""")

parser.add_argument("-g","--graspIdx",type=int,help="""Select Grasp index as indexed by DA-2 Dataset
                    default best grasp is at index 15
                    NOTE: Not all grasps are feasible or ideal""")

args = parser.parse_args()

if(args.model == "bimanual"):
    model_path = "/home/faizal/Documents/MuJoCo-Dual-Arm/models/bimanual_panda.xml";
else:
    model_path = "/home/faizal/Documents/MuJoCo-Dual-Arm/models/dual_panda.xml";

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

visFlag = 0

if(args.toggle_mocap == "False"):
    model.geom(model.body("targetL").geomadr).rgba = [0.0, 0.0, 0.0, 0.0]
    model.geom(model.body("targetR").geomadr).rgba = [0.0, 0.0, 0.0, 0.0]
    model.site(model.body("targetL").geomadr).rgba = [0.0, 0.0, 0.0, 0.0]
    model.site(model.body("targetR").geomadr).rgba = [0.0, 0.0, 0.0, 0.0]
    visFlag = 0
else:
    pass


Ipos = np.asarray([1000.0, 1000.0, 1000.0])  # [N/m]
Iori = np.asarray([100.0, 100.0, 100.0])  # [Nm/rad]
Kp_null = np.asarray([100.0, 100.0, 55.0, 55.0, 22.5, 20.0, 5.0, 2.0, 2.0,
                      100.0, 100.0, 55.0, 55.0, 22.5, 20.0, 5.0, 2.0, 2.0])
D = 1.3
Kpos = 6
Kori = 4

integration_dt = 0.1
gravity_compensation = True
dt = 0.002
# object_scale = 0.024724145342293464

def run(controller,tolerance, graspIdx):
    if(not tolerance):
       tolerance = 0.4
    if(not graspIdx):
        graspIdx = 15

    erL = 1000
    erR = 1000

    i = 0
    j = 0
    k = 0

    stage = 1

    grasps = np.load("examples/grasps/GraspChair.npy")
    graspL = grasps[graspIdx][0]
    graspR = grasps[graspIdx][1]
    objStrPos = [0.7,0.0,0.28]
    objStrOri = [-90,-90,0]
    # object_scale = 0.5
    # objStrPos = [0,0,0]
    # objStrOri = [0,0,0]
    object_scale = 1

    
    Util = RotationUtils()


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

    pre_grasp_pose_L = Util.GenPreGrasp(final_pose_L,0.22)
    pre_grasp_pose_R = Util.GenPreGrasp(final_pose_R,0.22)

    
    # pre_grasp_pose_L = ([-0.10, 0.33, 0.518],[0, 0, 1, 0]) 
    # pre_grasp_pose_R = ([0.03, 0.33, 0.518], [0, 1, 0, 0])

    DtrajL_pre = create_quintic_trajectory(init_pose_L, pre_grasp_pose_L, 1500)
    DtrajR_pre = create_quintic_trajectory(init_pose_R, pre_grasp_pose_R, 1500)

    
    jacP = controller.jac

    logger.info("INITIALISING TASK")
    while viewer.is_running():
        if(erL < tolerance and erR < tolerance and i <= 1500 and stage == 1):
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
            
        if(erL < tolerance and erR < tolerance and j <= 1500 and stage == 2):
            data.mocap_pos[controller.mocap_idL] = DtrajL[j][:3]
            data.mocap_quat[controller.mocap_idL] = DtrajL[j][3:]
            data.mocap_pos[controller.mocap_idR] = DtrajR[j][:3]
            data.mocap_quat[controller.mocap_idR] = DtrajR[j][3:]
            j += 1
            if(j == 1500):
                stage += 1
                logger.info("Initialising Stage Change: GRASP----->ASCEND")
                current_pose_L = (list(data.mocap_pos[controller.mocap_idL]), list(data.mocap_quat[controller.mocap_idL]))
                current_pose_R = (list(data.mocap_pos[controller.mocap_idR]), list(data.mocap_quat[controller.mocap_idR]))

                # init_object_pose = ([0.5, 0.0,0.3], [1, 0, 0, 1])
                init_object_pose = [list(data.body("collision_object").xpos.copy()),list(data.body("collision_object").xquat.copy())]
                final_object_pose = ([0.4,0.0,0.7], [1,0, 0, -1])

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
                logger.info("TASK COMPLETE")


 
        controller.armCtrl(jacP)

        erL = controller.erL
        erR = controller.erR

        if(stage == 1):
            controller.gripperCtrl("open", "both")
        elif(stage == 2):
            controller.gripperCtrl("open", "both")
        elif(stage == 3):
            controller.gripperCtrl("close", "both")
        elif(stage == 4):
            controller.gripperCtrl("open", "both")
            time.sleep(3)
            viewer.close()

        mujoco.mj_step(model, data)

        jacP = controller.jac

        viewer.sync()

def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Impedance(model, data, viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt, dt=dt,
                         gravity_compensation=gravity_compensation)
    
    controller.resetViewer(visFlag)

    run(controller,args.tolerance,args.graspIdx)

    # controller.makeplots()

if __name__ == "__main__":
    main()