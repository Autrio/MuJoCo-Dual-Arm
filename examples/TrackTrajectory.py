import mujoco 
import mujoco.viewer
import numpy as np
import time

from controllers.Impedance import Impedance

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

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Impedance(model,data,viewer)

    controller.setParams(Ipos=Ipos, Iori=Iori,
                         Kpos=Kpos, Kori=Kori, Kp_null=Kp_null,
                         D=D, integration_dt=integration_dt,dt=dt,
                         gravity_compensation=gravity_compensation)
    
    controller.resetViewer()
    
    while viewer.is_running():

        #set mocap pose to desired trajectory point for custom trajectory

        controller.armCrtl()
        mujoco.mj_step(model,data)
        viewer.sync()

if __name__ == "__main__":
    main()