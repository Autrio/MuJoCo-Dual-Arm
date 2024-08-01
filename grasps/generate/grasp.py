import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import grasps.generate.utils as utils
import h5py
import os
import torch

import argparse

parser = argparse.ArgumentParser(prog="grasp")

parser.add_argument("-o","--object",type=str,help="""Chose which Object to generate grasps for""")


args = parser.parse_args()

name = args.object

# functions for quaternion operations
def quat_mul(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(a)
    if not isinstance(b, torch.Tensor):
        b = torch.from_numpy(b)
        
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def normalize(x, eps: float = 1e-9):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_apply(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(a)
    if not isinstance(b, torch.Tensor):
        b = torch.from_numpy(b)
        
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def quat_conjugate(a):
    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(a)
        
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

def quat_inv(a):
    if not isinstance(a, torch.Tensor):
        a = torch.from_numpy(a)
        
    return quat_conjugate(a) / torch.sum(a ** 2, dim=-1, keepdim=True)

def R2T(rot=None, translation=None):
    T = np.eye(4)
    if rot is not None:
        T[:3, :3] = rot
    if translation is not None:
        T[:3, -1] = translation
    return T

obj_file = './grasps/objects/DA2-{}.obj'.format(name)
grasp_file = './grasps/DA-2-grasps/DA2-{}-Grasps.h5'.format(name)


grasp_org = h5py.File(grasp_file)
T_org = np.array(grasp_org['grasps/transforms'])
scale_org = grasp_org['object/scale'][()]
gripper = utils.create_gripper_marker()
T_org[0]

# tr1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), t1).matrix)
trimesh.load(obj_file).centroid

def plot_obj(obj_file, scale, do_mean_center=True, grasp_idx=10, current_T=None, next_T=None):
    grasp1, grasp2 = T_org[grasp_idx][0].copy(), T_org[grasp_idx][1].copy()
    grasp1[:, 3], grasp2[:, 3] = grasp1[:, 3], grasp2[:, 3]
    
    gripper_1 = gripper.copy().apply_transform(grasp1)
    gripper_2 = gripper.copy().apply_transform(grasp2)
    
    gripper_1.visual.face_colors = [255, 0, 0, 255]
    gripper_2.visual.face_colors = [0, 255, 0, 255]
    
    mesh = trimesh.load_mesh(obj_file)
    if do_mean_center:
        mesh.vertices -= mesh.center_mass
        
    mesh.apply_scale(scale)
    # mesh.export('monitor.obj')
    
    
    if current_T is not None:
        mesh.apply_transform(current_T)
        gripper_1.apply_transform(current_T)
        gripper_2.apply_transform(current_T)
        
    if next_T is not None:
        next_quat = R.from_matrix(next_T[:3, :3]).as_quat()
        mesh2 = mesh.copy()
        mesh2.apply_transform(next_T)
    
    frame = trimesh.creation.axis(axis_length=0.5)
    scene = trimesh.Scene([mesh, frame, gripper_1, gripper_2])
    if next_T is not None:
        scene.add_geometry(mesh2)
    return scene

if os.path.exists("/examples/generatedGrasps/grasp-{}.npy"):
    np.save('examples/generatedGrasps/grasp-{}.npy'.format(name), T_org)
    print(scale_org)
else:
    print(scale_org)