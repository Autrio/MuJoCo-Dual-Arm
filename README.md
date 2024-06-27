# MuJoCo-Dual-Arm
Franka Panda Bi-Manual Manipulation simulated on MuJoCo

## Visualising The Model
In your working directory,
```console
python3 -m mujoco.viewer --mjcf=/path/to/model/file.xml

```
## For Kinematics based Impedance Control:


```console
python3 -m examples.tests.controllerTest
```

**NOTE: Support DEPRECIATED**

## For Dynamic Task Space Control:
```console
python3 -m examples.tests.OpSpcTest
```
-Adapted from https://github.com/kevinzakka/mjctrl/tree/main