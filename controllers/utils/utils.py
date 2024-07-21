from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco

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
        
    
    def eefPose(self,data,eefname):
        eefPos = data.site(eefname).xpos
        eefQuat = np.zeros(4)
        mujoco.mju_mat2Quat(eefQuat,data.site(eefname).xmat)
        eefpose = np.concatenate((eefPos,eefQuat))
        return eefpose
    
    def Tmat2pose(self,mat,scale,objStrPos):
        if(not mat.shape == (4,4)):
            raise ValueError("matrix must be of shape 4x4 ")

        pos = (mat[:3,3:]).reshape(1,3)[0].tolist()
        pos += objStrPos
        rot = mat[:3,:3]
        rot = R.from_matrix(rot)
        quat = rot.as_quat()
        tempQuat = quat[1:]
        tempQuat = np.append(tempQuat,quat[0])

        return [pos,tempQuat.tolist()]

