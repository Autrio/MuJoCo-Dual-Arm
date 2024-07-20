import mujoco
import mujoco.viewer as mjv

import numpy as np
import time
from scipy.spatial.transform import Rotation as R


from controllers.utils.IK import GaussNewtonIK,GradientDescentIK,LevenbergMarquardtIK

model_path = "/home/autrio/college-linx/RRC/MuJoCo-Dual-Arm/models/dual_panda.xml";

model = mujoco.MjModel.from_xml_path(model_path);
model.opt.gravity = 0.0

data = mujoco.MjData(model);
 

body_id1 = model.site('end_effector').id
body_id2 = model.site('end_effector1').id
jac1 = np.zeros((6, model.nv)) #translation jacobian
jac2 = np.zeros((6, model.nv)) #rotational jacobian

r1 = [3.20041e-12, 0.923956, -0.382499, 1.32493e-12]
r2 = [3.20041e-12, 0.923956, -0.382499, 1.32493e-12]

r1 = R.from_quat(r1)
r2 = R.from_quat(r2)

r1 = r1.as_euler("xyz",degrees=False)
r2 = r2.as_euler("xyz",degrees=False)

goal1 = [0.30702, -0.75, 0.59027]
goal2 = [0.30702, 0.75, 0.59027]

goal1.extend(r1)
goal2.extend(r2)

step_size = 0.001
tol = 0.5
alpha = 0.01
init_q = data.qpos[:18]
viewer = mjv.launch_passive(model,data,show_left_ui=False,
        show_right_ui=False)

ik = GaussNewtonIK(model, data, step_size, tol, alpha, jac1, jac2,viewer)


#Get desire point
mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
traj  = ik.solve(goal1, goal2, init_q, body_id1,body_id2) #calculate the qpos

f = open("./examples/tests/TRAJECTORY_LOG.txt","w");
for qpos in ik.trajectory:
    f.write(str(qpos)+"\n")
f.close();


mujoco.mj_resetDataKeyframe(model, data, 0) #reset qpos to initial value
mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value

time.sleep(2)
viewer.close();

print(len(traj[-1]))



