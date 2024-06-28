import mujoco 
import mujoco.viewer
import numpy as np
import time
import argparse as ap

from controllers.Impedance import Impedance

parser = ap.ArgumentParser(prog="pnp",
                           description="simple pick and place task")

parser.add_argument("-d","--model",type=str,help="""Choose variant of dual panda arms.
                    'dual' for individual separate arms, 'bimanual' for arms connected
                     to a torso at shoulder joint. Default is 'dual'""")

args = parser.parse_args()


if(args.model == "bimanual"):
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/bimanual_panda.xml";
else:
    model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";

model = mujoco.MjModel.from_xml_path(model_path);
data = mujoco.MjData(model);
viewer = mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False)

Ipos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
Iori = np.asarray([200.0, 200.0, 200.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([10.0, 10.0, 5.0, 5.0, 4.0, 2.50, 2.50, 1.0, 1.0,
                      10.0, 10.0, 5.0, 5.0, 4.0, 2.50, 2.50, 1.0, 1.0])

# Damping ratio for both Cartesian and joint impedance control.
D = 1.3

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.8

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 1

# Integration timestep in seconds.
integration_dt: float = 0.1

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

def makecircle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y])

def LineartrajectoryZ(t, startPose, finalZ):
    traj = []
    Ztrail = np.linspace(startPose[2],finalZ,t)
    for pnt in Ztrail:
        traj.append([startPose[0],startPose[1],pnt])
    
    return traj

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Impedance(model,data,viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt,dt=dt,
                         gravity_compensation=gravity_compensation)
    
    controller.resetViewer()

    tolerance = 0.02
    erL = 1000
    erR = 1000

    i = 0
    j = 0

    stage = 1

    DtrajL = LineartrajectoryZ(1500,data.mocap_pos[controller.mocap_idL],0.268);
    DtrajR = LineartrajectoryZ(1500,data.mocap_pos[controller.mocap_idR],0.268);

    while viewer.is_running():
        #set mocap pose to desired trajectory point for custom trajectory
        if(erL<tolerance and erR < tolerance and i<1500 and stage==1):
            data.mocap_pos[controller.mocap_idL] = DtrajL[i]
            data.mocap_pos[controller.mocap_idR] = DtrajR[i]
            i+=1
            if(i==1500):
                stage+=1
                AtrajL = LineartrajectoryZ(1500,data.mocap_pos[controller.mocap_idL],0.5);
                AtrajR = LineartrajectoryZ(1500,data.mocap_pos[controller.mocap_idR],0.5);


        if(erL<tolerance and erR < tolerance and j<1500 and stage==2):
            data.mocap_pos[controller.mocap_idL] = AtrajL[j]
            data.mocap_pos[controller.mocap_idR] = AtrajR[j]
            j+=1
            if(j==1500):
                stage+=1

        controller.armCrtl()

        erL = np.linalg.norm(controller.dxL)
        erR = np.linalg.norm(controller.dxR)

        if(stage==1):
            controller.gripperCtrl("open","both")
        elif(stage==2):
            controller.gripperCtrl("close","both")
        elif(stage==3):
            controller.gripperCtrl("open","both")
            # time.sleep(1)
            viewer.close()

        mujoco.mj_step(model,data)
        viewer.sync()

if __name__ == "__main__":
    main()