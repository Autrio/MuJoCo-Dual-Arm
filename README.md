# MuJoCo-Dual-Arm
Franka Panda Bi-Manual Manipulation simulated on MuJoCo

## Large Object manipulation

https://github.com/user-attachments/assets/f984b51f-ebf6-4123-88dc-18e3e2745001

## Visualising The Model
In the repository directory ,
```console
python3 -m mujoco.viewer --mjcf=/full/path/to/model/file.xml
```

## Model Select Parameter:
- `-d dual` or `--model dual` for Two separated Panda arms 
- `-d bimanual` or `--model bimanual` for Two Panda Arms connected at shoulder Joints to a torso
- `-v False` or `--toggle-mocap False` to turn off frame and error visualisation (mocap bodies)
- `-t <FLOAT>` to set maximum trajectory following error



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
without frame visuals
```console
python3 -m examples.pnp -d dual -v False
```

with different error tolerance
```console
python3 -m examples.pnp -d dual -t 0.1 
```

**NOTE : Experimental** 

**TODO** : 
- Test for different objects
- Adjust to optimal gain parameters
- Trajectory optimisation and motion planning
