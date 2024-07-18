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
jacp = np.zeros((6, model.nv)) #translation jacobian
jacr = np.zeros((6, model.nv)) #rotational jacobian

r1 = [0 ,0 ,1 ,0]
r2 = [0 ,0 ,1 ,1]

r1 = R.from_quat(r1)
r2 = R.from_quat(r2)

r1 = r1.as_euler("xyz",degrees=False)
r2 = r2.as_euler("xyz",degrees=False)

goal1 = [-0.0444176,-0.249729, 0.32524]
goal2 = [-0.0395817, 0.259253,0.306346]

goal1.extend(r1)
goal2.extend(r2)

step_size = 0.001
tol = 0.05
alpha = 0.1
init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
viewer = mjv.launch_passive(model,data)

ik = GaussNewtonIK(model, data, step_size, tol, alpha, jacp, jacr,viewer)


#Get desire point
mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
ik.solve(goal1, goal2, init_q, body_id1,body_id2) #calculate the qpos

f = open("./examples/tests/TRAJECTORY_LOG.txt","w");
for qpos in ik.trajectory:
    f.write(str(qpos)+"\n")
f.close();


mujoco.mj_resetDataKeyframe(model, data, 0) #reset qpos to initial value
mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value

time.sleep(1);
viewer.close();



