import mujoco
import mujoco.viewer as mjv

import numpy as np
import time


from controllers.utils.IK import GaussNewtonIK,GradientDescentIK,LevenbergMarquardtIK

model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";

model = mujoco.MjModel.from_xml_path(model_path);

data = mujoco.MjData(model);

renderer = mujoco.Renderer(model)

body_id1 = model.body('right_panda_gripper').id
body_id2 = model.body('left_panda_gripper').id
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian
goal1 = [0.49, 0.13, 0.59]
goal2 = [-0.49, 0.13, 0.59]
step_size = 0.1
tol = 0.05
alpha = 0.5
init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
viewer = mjv.launch_passive(model,data)

ik = GaussNewtonIK(model, data, step_size, tol, alpha, jacp, jacr,viewer)

#Get desire point
mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
ik.solve(goal1, init_q, body_id1) #calculate the qpos
ik.solve(goal2, init_q, body_id2) #calculate the qpos



# mujoco.mj_resetDataKeyframe(model, data, 0) #reset qpos to initial value
# mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value



time.sleep(5);
viewer.close();
exit(0);



