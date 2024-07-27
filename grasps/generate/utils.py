# Copyright (c) 2021 Jens Lundell
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import copy
import glob
import multiprocessing
import os

import h5py
import numpy as np
import open3d as o3d
import trimesh
import trimesh.transformations as tra
from joblib import Parallel, delayed
from scipy.spatial import distance_matrix


def return_files_in_folder(path, extension=""):
    abspath = os.path.abspath(path) + "/"
    files = glob.glob(abspath + "*" + extension)
    return files


def load_mesh_from_file(mesh_file, scale=1):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
    mesh.paint_uniform_color([0.768, 0.768, 0.768])
    return mesh


def create_folder(folder):
    try:
        os.makedirs(folder)
    except FileExistsError:
        print("Directory ", folder, " already exists")


def visualize_scene(object_mesh, object_point_cloud, gripper=None):
    o3d_point_cloud = create_o3d_point_cloud(object_point_cloud)
    if type(gripper) is list:
        geometries_to_draw = [object_mesh, o3d_point_cloud] + gripper
        o3d.visualization.draw_geometries(geometries_to_draw)
    else:
        o3d.visualization.draw_geometries([object_mesh, o3d_point_cloud])


def create_o3d_point_cloud(point_cloud, color=[0, 0, 1]):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    point_cloud_o3d.paint_uniform_color(color)
    return point_cloud_o3d


def get_basename_from_file(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return basename


def find_match(list_to_find_match, string_for_matching):
    for l in list_to_find_match:
        if string_for_matching in l:
            return l
    return ""


def read_h5_file(file):
    data = h5py.File(file, "r")
    return data


# def create_gripper_marker(color=[0, 0, 1], sections=6):
#     """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.
#     Args:
#         color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
#         tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
#         sections (int, optional): Number of sections of each cylinder. Defaults to 6.
#     Returns:
#         trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
#     """
#     cfl = trimesh.creation.cylinder(
#         radius=0.002,
#         sections=sections,
#         height=0,
#         segment=[
#             [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
#             [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
#         ],
#     )
#     cfr = trimesh.creation.cylinder(
#         radius=0.002,
#         sections=sections,
#         height=0,
#         segment=[
#             [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
#             [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
#         ],
#     )
#     cb1 = trimesh.creation.cylinder(radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]], height=0)
#     cb2 = trimesh.creation.cylinder(
#         radius=0.002, sections=sections, segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]], height=0
#     )
#     gripper_mesh = trimesh_to_open3d(cfl)
#     gripper_mesh += trimesh_to_open3d(cfr)
#     gripper_mesh += trimesh_to_open3d(cb1)
#     gripper_mesh += trimesh_to_open3d(cb2)
#     # gripper_mesh.color = color
#     gripper_mesh = gripper_mesh.paint_uniform_color(color)

#     return gripper_mesh


def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 0],
            [4.10000000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 0],
            [-4.100000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, -0.067500/2], [0, 0, 0]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-0.085/2, 0, 0], [0.085/2, 0, 0]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    R = np.array([[1,0,0],[0,0,1],[0,-1,0]]).reshape(3,3)
    t =  np.array([0, 0, 0]).reshape(3,1)
    #
    T = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
    tmp.apply_transform(T)

    return tmp




def trimesh_to_open3d(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    return o3d_mesh


def get_grasp_center_point():
    # right_gripper_top = np.asarray([[4.10000000e-02, -7.27595772e-12, 1.12169998e-01]])
    # left_gripper_top = np.asarray([[-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]])
    # right_gripper_base = np.asarray([[4.10000000e-02, -7.27595772e-12, 6.59999996e-02]])
    # left_gripper_base = np.asarray([[-4.10000000e-02, -7.27595772e-12, 6.59999996e-02]])
    
    right_gripper_top = np.asarray([[0.041, 0.0675, 0]])
    right_gripper_base = np.asarray([[0.041, 0, 0]])
    left_gripper_top = np.asarray([[-0.041, 0.0675, 0]])
    left_gripper_base = np.asarray([-0.041, 0, 0])

    center_point = (right_gripper_top + left_gripper_top + right_gripper_base + left_gripper_base) / 4.0
    return center_point


def setup_gripper():
    gripper_marker = create_gripper_marker()
    grasp_center_point = get_grasp_center_point()
    return gripper_marker, grasp_center_point


def transform_vector(vector, transformation_matrix=np.eye(4)):
    new_vector = np.ones((vector.shape[0], 4))
    new_vector[:, :3] = vector
    transformed_vector = transformation_matrix.dot(new_vector.T).squeeze()
    if transformed_vector.shape[0] == 4:
        transformed_vector = transformed_vector[:3, :].T
    elif transformed_vector.shape[1] == 4:
        transformed_vector = transformed_vector[:, :3]
    return transformed_vector


def euclid_dist(list_a, list_b):
    distances = distance_matrix(list_a, list_b)
    return distances


def save_dataset(dataset, file_name, path):
    dt = h5py.vlen_dtype(np.dtype("int32"))
    remove = False
    if not file_exists(path + "constrained_" + file_name + ".h5"):
        with h5py.File(path + "constrained_" + file_name + ".h5", "w") as f:
            for key in dataset.keys():
                if (
                    key == "query_points/points_with_grasps_on_each_rendered_point_cloud"
                    or key == "query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"
                ):
                    try:
                        print("Saving data")
                        f.create_dataset(key, 
                                         data=np.array(dataset[key], dtype=object), 
                                         dtype=h5py.special_dtype(vlen=np.dtype('object')))
                        
                    except Exception as e:
                        print(e)
                        input()
                        # remove = True
                        # continue
                else:
                    f.create_dataset(key, data=dataset[key])
    if remove:
        os.remove(path + "constrained_" + file_name + ".h5")


def file_exists(file_path):
    return os.path.exists(file_path)


def get_uniformly_spaced_orientations():
    all_poses = []
    for az in np.linspace(0, np.pi * 2, 5):
        for el in np.linspace(-np.pi / 2, np.pi / 2, 5):
            all_poses.append(tra.euler_matrix(el, az, 0))
    return all_poses


def regularize_pc_point_count(pc, npoints):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        # if use_farthest_point:
        #    _, center_indexes = farthest_points(pc,
        #                                        npoints,
        #                                        distance_by_translation_point,
        #                                        return_center_indexes=True)
        # else:
        center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def visualize_all_grasps_per_query_point(
    mesh, object_point_cloud, query_point_with_grasps, gripper_visualizer, grasp_transformation_matrices, grasp_indices, camera_pose
):
    all_grippers = []
    for grasp_index in grasp_indices:
        gripper_per_index = copy.deepcopy(gripper_visualizer)
        gripper_per_index.transform(grasp_transformation_matrices[grasp_index])
        gripper_per_index.transform(camera_pose)
        all_grippers.append(gripper_per_index)
    visualize_scene(mesh, object_point_cloud, query_point_with_grasps, all_grippers)


def visualize_scene(mesh, object_point_cloud, points_with_grasps, gripper=None):
    o3d_point_cloud = create_o3d_point_cloud(object_point_cloud, color=[0, 1, 0])
    if points_with_grasps.size > 1:
        for point_with_grasp in points_with_grasps:
            o3d_point_cloud.colors[point_with_grasp] = [1, 0, 0]
    else:
        o3d_point_cloud.colors[points_with_grasps] = [1, 0, 0]
    if type(gripper) is list:
        geometries_to_draw = [mesh, o3d_point_cloud] + gripper
        o3d.visualization.draw_geometries(geometries_to_draw)
    else:
        o3d.visualization.draw_geometries([mesh, o3d_point_cloud])


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


def cluster_points(points, num_clusters, threshold_for_neighbors, max_points_per_cluster=1024):
    farthest_points, _ = farthest_point_sample(points, num_clusters)
    neighbors_to_farthest_points = find_neighboring_points(farthest_points, points, threshold_for_neighbors, max_points_per_cluster)
    return neighbors_to_farthest_points


def remove_clusters_with_no_grasps(clusters, grasp_indices):
    points_with_grasps = np.unique(grasp_indices[:, 0])
    clusters_to_remove = []
    for cluster_idx, cluster in enumerate(clusters):
        if np.isin(cluster, points_with_grasps).sum() == 0:
            clusters_to_remove.append(cluster_idx)
    for i in sorted(clusters_to_remove, reverse=True):
        del clusters[i]
    return clusters


def find_neighboring_points(points, query_points, distance_threshold_for_neighbors=0.01, max_num_neighbors=1024):
    distances = euclid_dist(points, query_points)
    indexes = distances < distance_threshold_for_neighbors
    neighbors_per_cluster = []
    for i in range(indexes.shape[0]):
        neighbors = np.argwhere(indexes[i]).squeeze(axis=-1)
        if neighbors.size > max_num_neighbors:
            center_indexes = np.random.choice(range(neighbors.shape[0]), size=max_num_neighbors, replace=False)
            neighbors = neighbors[center_indexes]
        neighbors_per_cluster.append(neighbors)
    return neighbors_per_cluster


def farthest_point_sample(point, npoint=1000):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroid = centroids.astype(np.int32)
    farthest_points = point[centroids.astype(np.int32)]
    points_left = np.delete(point, centroid, 0)
    return farthest_points, points_left


def get_grasps_per_cluster_parallel(clusters, grasp_indices):
    grasps_per_cluster = []

    def temp(cluster, grasp_indices_copy):
        grasps_for_one_cluster = []
        for point_index in cluster:
            rows_for_current_query_points = grasp_indices_copy[:, 0] == point_index
            grasp_indices_at_current_query_point = grasp_indices_copy[rows_for_current_query_points, 1]
            grasps_for_one_cluster.append(grasp_indices_at_current_query_point)
            grasp_indices_copy = np.delete(grasp_indices_copy, rows_for_current_query_points, 0)
        return np.unique(np.concatenate(grasps_for_one_cluster))

    num_cores = multiprocessing.cpu_count() / 2
    grasps_per_cluster = Parallel(n_jobs=int(num_cores))(delayed(temp)(cluster, grasp_indices) for cluster in clusters)
    return grasps_per_cluster


def radius_of_sphere_that_circumscribes_the_mesh(o3d_mesh):
    mesh_bounding_box = o3d_mesh.get_axis_aligned_bounding_box()
    mesh_bounding_box_center = mesh_bounding_box.get_center()
    mesh_bounding_box_one_corner = np.asarray(mesh_bounding_box.get_box_points())[0]
    radius_of_circumscribing_sphere = np.linalg.norm(mesh_bounding_box_center - mesh_bounding_box_one_corner)
    return radius_of_circumscribing_sphere


def plot_pc_grasp(pc, gripper_pc = None, T = [], success=None, n=10, mask=None):
    # Red, Blue, Green, Black
    colors = np.array([[255, 0, 0], [0, 0, 255.0], [0, 255.0, 0], [0, 0, 0]]).reshape(-1, 1, 3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if mask is not None:
        color = np.repeat(colors[1], len(pc), axis=0)
        color[mask] = np.repeat(colors[0], len(mask), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
    else: 
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(colors[1], len(pc), axis=0))
    
    grippers = []
    
    if gripper_pc is not None:
        indices = np.random.choice(len(T), n, replace=False)
        
        T = T[indices]
      
        for i, t in enumerate(T):
            gripper = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_pc))  
            gripper_temp = gripper.transform(t)
            
            gripper_temp.colors = o3d.utility.Vector3dVector(np.repeat(colors[2], len(np.asarray(gripper.points)), axis=0))
            grippers.append(gripper_temp)
        
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    visualizer.add_geometry(pcd)
    visualizer.add_geometry(coordinate_frame)
    
    for gripper_pcd in grippers:
        visualizer.add_geometry(gripper_pcd)

    visualizer.run()
    
def load_obj_and_convert_to_pc(obj_file_path, n=1024):
    # Load the OBJ file
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    # Convert the mesh to a PointCloud
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=n)
    return np.asarray(point_cloud.points)

def mesh2pc(mesh, n=4096):
    # vertices, faces = mesh.vertices, mesh.faces
    # triangulated_mesh = o3d.geometry.TriangleMesh()
    # triangulated_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # triangulated_mesh.triangles = o3d.utility.Vector3iVector(faces)

    point_cloud = mesh.sample_points_uniformly(number_of_points=n)

    return np.array(point_cloud.points)

def mean_center_pc(pc):
    mean = pc.mean(0)
    pc = pc - mean
    return pc, mean

def mean_center_T(T, mean):
    T[:, :3, 3] -= mean
    return T

def transform_pc(pc, T, center_it=False):
    ones = np.ones((len(pc), 1))
    pc = np.hstack((pc, ones))
    pc = (pc @ T.T)[:, :3]
    if center_it:
        pc, _ = mean_center_pc(pc)
    return pc


def plot_pcs_together(pcs):
    colors = np.array([[255, 0, 0], [0, 0, 255.0], [0, 255.0, 0], [0, 0, 0]]).reshape(-1, 1, 3)
    pcds = []
    
    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(colors[i], len(pc), axis=0))
        pcds.append(pcd)
        
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    visualizer.add_geometry(coordinate_frame)
    for pcd in pcds:
        visualizer.add_geometry(pcd)
    
    visualizer.run()
    