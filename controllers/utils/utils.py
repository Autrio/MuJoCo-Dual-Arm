from scipy.spatial.transform import Rotation as R
import numpy as np

class RotationUtils:
    def __init__(self) -> None:
        pass

    def Quat2rot(self,quat,informat:str,outformat:str,degrees:bool):
        if(informat=="xyzw"):
            r = R.from_quat(quat)
            return(r.as_euler(outformat,degrees))
        elif(informat=="wxyz"):
            tempQuat = quat[1:]
            tempQuat = np.append(tempQuat,quat[0])
            r = R.from_quat(tempQuat)
            return(r.as_euler(outformat,degrees))
        else:
            raise(ValueError)