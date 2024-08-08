
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
    
    def Tmat2pose(self,mat,scale,objStrPos,objStrOri):
        if(not mat.shape == (4,4)):
            raise ValueError("matrix must be of shape 4x4 ")

        pos = (mat[:3,3:]*scale).reshape(1,3)[0]
        pos += objStrPos
        rot = mat[:3,:3]
        rotTf = R.from_euler("xyz",(objStrOri[0],objStrOri[1],objStrOri[2]),degrees=True)
        rotTform = rotTf.as_matrix()
        rot = rot @ rotTform
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat,rot.flatten())

        return [pos.tolist(),quat.tolist()]

    def GenPreGrasp(self,grasp,offset):
        x, y, z = grasp[0]
        qw ,qx ,qy ,qz = grasp[1]

        rot = R.from_quat([qx,qy,qz,qw])
        Rmat = rot.as_matrix()

        graspAxis = Rmat[:,1]

        offsetPos = np.array([x,y,z]) - offset*graspAxis

        preGraspPose = [offsetPos.tolist(),[qw,qx,qy,qz]]
        return preGraspPose

    def safe_matrix_sqrt(self,matrix):
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Take square root of the absolute values of the eigenvalues
        sqrt_eigvals = np.sqrt(np.abs(eigvals))
        
        # Reconstruct the square root of the matrix
        sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
        
        return sqrt_matrix

    def compute_damping_matrices(self, MxL, MxR, K):
        # Compute the square roots using the safe square root function
        sqrt_MxL = self.safe_matrix_sqrt(MxL)
        sqrt_MxR = self.safe_matrix_sqrt(MxR)
        sqrt_K = np.sqrt(K)  # K has no negative eigenvalues
        
        # Compute the damping matrices
        DL = sqrt_MxL @ sqrt_K + sqrt_K @ sqrt_MxL
        DR = sqrt_MxR @ sqrt_K + sqrt_K @ sqrt_MxR

        return DL, DR
