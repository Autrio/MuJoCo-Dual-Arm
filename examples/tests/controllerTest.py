import mujoco
import mujoco.viewer as viewer

from controllers.Impedance import Impedance

model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";
model = mujoco.MjModel.from_xml_path(model_path);
data = mujoco.MjData(model);

impControl = Impedance(model,data,"right_panda_gripper",[-3.14,3.14])