import mujoco
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
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

integration_dt = 0.1
gravity_compensation = True
dt = 0.002

class Convex_Params:
    def __init__(self,model,data,viewer) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer

    def setParams(self,integration_dt:float,gravity_compensation:bool,
                  dt:float,site_nameL="end_effector",site_nameR="end_effector1"):
        
        self.integration_dt = integration_dt
        self.gravity_compensation = gravity_compensation
        self.dt = dt
        self.site_nameL = site_nameL
        self.site_nameR = site_nameR

        self.model.opt.timestep = self.dt

        # Compute damping and stiffness matrices.
        self.damping_pos = self.D * 2 * np.sqrt(self.Ipos)
        self.damping_ori = self.D * 2 * np.sqrt(self.Iori)
        self.Kp = np.concatenate([self.Ipos, self.Iori], axis=0)
        self.Kd = np.concatenate([self.damping_pos, self.damping_ori], axis=0)
        self.Kd_null = self.D * 2 * np.sqrt(self.Kp_null)

        # End-effector site we wish to control.
        self.site_idL = self.model.site(self.site_nameL).id

        self.site_idR = self.model.site(self.site_nameR).id

        # Get the dof and actuator ids for the joints we wish to control. These are copied
        # from the XML file. Feel free to comment out some joints to see the effect on
        # the controller.
        self.joint_names = [self.model.jnt(name).name for name in range(self.model.njnt)]

        self.actuator_names = [self.model.actuator(name).name for name in range(self.model.njnt-1)]


        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])


        self.dof_idsL = self.dof_ids[:9]
        self.dof_idsR = self.dof_ids[9:18]

        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.actuator_names])

        self.actuator_idsL = self.actuator_ids[:9]
        self.actuator_idsR = self.actuator_ids[9:]

        # Initial joint configuration saved as a keyframe in the XML file.
        self.key_name = "home"
        self.key_id = self.model.key(self.key_name).id
        self.q0 = self.model.key(self.key_name).qpos[:18]
        self.qd0 = self.data.qvel[:18]

        # Mocap body we will control with our mouse.
        self.mocap_nameL = "targetL"
        self.mocap_idL = self.model.body(self.mocap_nameL).mocapid[0]

        self.mocap_nameR = "targetR"
        self.mocap_idR = self.model.body(self.mocap_nameR).mocapid[0]

        # Pre-allocate numpy arrays.
        self.jacR = np.zeros((6, self.model.nv))
        self.jacL = np.zeros((6, self.model.nv))
        self.jac = np.zeros((6, 18)) # the jacobian for the arms
        self.jacPrev = np.zeros((6,18)) #prev values of jac for finite difference jdot

    
        self.M_all = np.zeros((self.model.nv, self.model.nv))

        self.Mx = np.zeros((6, 6))

        self.eye = np.eye(18)

        self.twistL = np.zeros(6)
        self.twistR = np.zeros(6)

        self.site_quatL = np.zeros(4)
        self.site_quatR = np.zeros(4)

        self.site_quat_conjL = np.zeros(4)
        self.site_quat_conjR = np.zeros(4)

        self.error_quatL = np.zeros(4)
        self.error_quatR = np.zeros(4)

        #arrays for plotting
        self.setupDatacap()
        
        
    def resetViewer(self,flag):
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

        # Enable site frame visualization.
        if(flag):
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE    
            
    
    def conv_parameters(self,JacP):
        self.jacPrev = JacP
        M = self.data.qM
        C = self.data.qfrc_bias
        J = self.data.efc_J[self.dof_ids[:18]]
        J_dot = (J - self.jacPrev)/self.integration_dt 
        
        print("Mass - Inertia Matrix : ",M)
        print("Coriolis and Gravity : ",C)
        print("Jacobian : ",J)  
        print("Jacobian_dot : ",J_dot)
        

        return M,C,J,J_dot
    
            

def run(controller,tolerance,graspIdx):
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

    controller = Convex_Params(model, data, viewer)

    controller.setParams(integration_dt=integration_dt, dt=dt,
                         gravity_compensation=gravity_compensation)
    
 

    run(controller,tolerance=args.tolerance,graspIdx=args.graspIdx)



if __name__ == "__main__":
    main()
    
    