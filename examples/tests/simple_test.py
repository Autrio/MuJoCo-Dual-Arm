import mujoco as mj
import mujoco.viewer as mjv
import mediapy as media
import time

import numpy as np

model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";

robot = mj.MjModel.from_xml_path(model_path);

data = mj.MjData(robot);
names = [robot.jnt(i).name for i in range(robot.njnt)];

viewer = mjv.launch_passive(robot,data);
while(viewer.is_running()):
        
    # print(data.site("end_effector1").xpos);
    mj.mj_step(robot,data);
    viewer.sync();
    # time.sleep(1);

# viewer.close();
exit(0);