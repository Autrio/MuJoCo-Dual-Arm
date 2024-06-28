# MuJoCo-Dual-Arm
Franka Panda Bi-Manual Manipulation simulated on MuJoCo

## Visualising The Model
In the repository directory directory,
```console
python3 -m mujoco.viewer --mjcf=/full/path/to/model/file.xml
```

## Model Select Parameter:
- `-d dual` or `--model dual` for Two separated Panda arms 
- `-d bimanual` or `--model bimanual` for Two Panda Arms connected at shoulder Joints to a torso


## For Kinematics based Impedance Control:
```console
python3 -m examples.tests.JntSpcTest -d dual
```

**NOTE: Support DEPRECIATED**

## For Dynamic Task Space Control:
```console
python3 -m examples.tests.OpSpcTest -d dual
```

## For Pick and Place Task:
```console
python3 -m examples.pnp -d dual
```
**NOTE : Experimental** 

**TODO** : 
- Fix gripper command and actuation to prevent slipping
- Adjust to optimal gain parameters
- Trajectory optimisation and motion planning

-Adapted from https://github.com/kevinzakka/mjctrl/tree/main