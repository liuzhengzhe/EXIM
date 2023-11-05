import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
from utils.other_utils import sample_points_triangles, load_off, write_ply_point_normal, load_obj
import h5py
import numpy as np
from utils.debugger import MyDebugger
import multiprocessing as mp
from trimesh import Trimesh
from trimesh.sample import sample_surface
from scipy.io import loadmat

from mesh_to_sdf import sample_sdf_near_surface
import pyrender
import trimesh
import traceback
from itertools import compress
import os
from typing import Tuple
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def compute_unit_sphere_transform(mesh: Trimesh) -> Tuple[np.ndarray, float]:
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

def sample_pointsnormals_trimesh(vertices, faces, sample_num):
    mesh = Trimesh(vertices=vertices, faces=faces)
    positions, faces_indices = sample_surface(mesh, sample_num)
    sampled_normals = mesh.face_normals[faces_indices]
    samples = np.concatenate((positions, sampled_normals), axis=1)

    return samples


def sample_data(file_paths, voxels_paths, output_file, pc_number_of_points=50000, sdf_number_of_points = 500000, normalize=False, save_ply=False,
                save_folder=None, view_sdf = False, workers = 12, indices = None, split_data = False):

    if not split_data:
        f = h5py.File(output_file, "w")

    pool = mp.Pool(workers)

    args = [(file_path, voxels_paths[idx], normalize, pc_number_of_points, sdf_number_of_points, save_folder, save_ply, view_sdf, split_data) for idx, file_path in
            enumerate(file_paths)]
    if indices is not None:
        args = args[:indices]
    # for arg in args:
    #     save_one_pointsnormals(arg)

    samples = pool.map(save_one_pointsnormals, args)
    point_normals, points, sdfs, valids = zip(*samples)
    point_normals, points, sdfs, valids = list(point_normals), list(points), list(sdfs), list(valids)
    point_normals, points, sdfs = list(compress(point_normals, valids)), list(compress(points, valids)), list(compress(sdfs, valids))
    point_normals, points, sdfs = np.array(point_normals), np.array(points), np.array(sdfs)

    if not split_data:
        f.create_dataset('pointnormals', data=point_normals)
        f.create_dataset('sdfs', data=sdfs)
        f.create_dataset('points', data=points)
        f.close()


def save_one_pointsnormals(args):
    try:
        sample_ratio = 1.5
        file_path, voxel_path, normalize, pc_number_of_points, sdf_number_of_points, save_folder, save_ply, view_sdf, split_data = args
        sample_number = int(pc_number_of_points * sample_ratio)
        # vertices, _, faces = load_off(file_path + '.off')
        vertices, _, faces, _ = load_obj(file_path + '/model_flipped_manifold.obj')

        ### voxel
        sample_mat = loadmat(voxel_path)
        occ = get_voxels(sample_mat, save_path=os.path.join(save_folder, os.path.basename(file_path) + "_voxels.xyz"),
                         save_points=False)


        ### sdf
        mesh = trimesh.load_mesh(file_path + '/model_flipped_manifold.obj')
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=sdf_number_of_points)

        translation, scale = compute_unit_sphere_transform(mesh)
        points = (points / scale) - translation
        sdf /= scale

        if view_sdf:
            colors = np.zeros(points.shape)
            colors[sdf < 0, 2] = 1
            colors[sdf > 0, 0] = 1
            cloud = pyrender.Mesh.from_points(points, colors=colors)
            scene = pyrender.Scene()
            scene.add(cloud)
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


        if normalize:
            mean_pos = np.mean(vertices, axis=0)
            vertices = vertices - mean_pos
            print(f"mean of vertices {mean_pos}!")

        number_sampled, cnt = 0, 0
        sample = np.zeros((0, 6))
        while (sample.shape[0] < pc_number_of_points and cnt < 5):
            # new_sample = sample_points_triangles(num_of_points = sample_number, triangles = faces, vertices = vertices)
            new_sample = sample_pointsnormals_trimesh(vertices, faces, sample_num=sample_number)

            ### compute occupancy
            vertices_in_voxel = np.array(np.floor((new_sample[:, :3] + 0.5) * 256) - 0.5, dtype=np.uint8)
            vertices_on_boundary_max = np.zeros(vertices_in_voxel.shape[0], dtype=np.uint8)
            vertices_on_boundary_min = np.zeros(vertices_in_voxel.shape[0], dtype=np.uint8)
            normal_flipped = np.zeros(vertices_in_voxel.shape[0], dtype=np.uint8)

            for idx in range(vertices_in_voxel.shape[0]):
                diff = 2
                testing_range = occ[
                                max(vertices_in_voxel[idx, 0] - diff, 0): min(vertices_in_voxel[idx, 0] + diff + 1, 256),
                                max(vertices_in_voxel[idx, 1] - diff, 0): min(vertices_in_voxel[idx, 1] + diff + 1, 256),
                                max(vertices_in_voxel[idx, 2] - diff, 0): min(vertices_in_voxel[idx, 2] + diff + 1, 256)]
                x_dim = np.arange(max(vertices_in_voxel[idx, 0] - diff, 0), min(vertices_in_voxel[idx, 0] + diff + 1, 256))
                y_dim = np.arange(max(vertices_in_voxel[idx, 1] - diff, 0), min(vertices_in_voxel[idx, 1] + diff + 1, 256))
                z_dim = np.arange(max(vertices_in_voxel[idx, 2] - diff, 0), min(vertices_in_voxel[idx, 2] + diff + 1, 256))
                x, y, z = np.meshgrid(x_dim, y_dim, z_dim)
                x, y, z = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis], z[:, :, :, np.newaxis]
                coordinates = np.concatenate((x, y, z), axis=3).reshape((x_dim.shape[0], y_dim.shape[0], z_dim.shape[0], 3))

                ## inside coordinates
                inside_coordinates = coordinates * testing_range[:, :, :, np.newaxis] / np.sum(testing_range) if np.sum(
                    testing_range) > 0 else np.array([0, 0, 0])
                outside_coordinates = coordinates * (1 - testing_range[:, :, :, np.newaxis]) / np.sum(
                    1 - testing_range) if np.sum(1 - testing_range) > 0 else np.array([0, 0, 0])

                vertices_on_boundary_max[idx] = np.max(testing_range)
                vertices_on_boundary_min[idx] = np.min(testing_range)
                # max_coord = ((np.array(np.unravel_index(np.argmin(testing_range), testing_range.shape)) + vertices_in_voxel[idx] - diff) + 0.5) / 256 - 0.5
                # min_coord = (( np.array(np.unravel_index(np.argmax(testing_range), testing_range.shape)) + vertices_in_voxel[idx] - diff) + 0.5) / 256 - 0.5
                estimate_normal = outside_coordinates - inside_coordinates
                normal_flipped[idx] = np.sum(estimate_normal * new_sample[idx, 3:]) < 0

            ### sample
            # on_surface_flag = vertices_on_boundary_max != vertices_on_boundary_min
            on_surface_flag = vertices_on_boundary_min == 0
            # on_surface_flag = np.ones(sample_number) == 1
            new_sample[normal_flipped, 3:] = -new_sample[normal_flipped, 3:]
            new_sample = new_sample[on_surface_flag]
            sample = np.concatenate((sample, new_sample), axis=0)
            print(f"Rejected sample of {file_path} = {np.sum(1 - on_surface_flag)}")

        sample = sample[:pc_number_of_points]

        if save_ply and save_folder is not None:
            xyz_path = os.path.join(save_folder, os.path.basename(file_path) + ".xyz")
            xyz_points_path = os.path.join(save_folder, os.path.basename(file_path) + "_points_samples.xyz")
            print(f"{xyz_path} saved!")
            np.savetxt(xyz_path, sample)
            np.savetxt(xyz_points_path, points)
            # write_ply_point_normal(ply_path, vertices = sample[:, :3], normals=sample[:, 3:])
        print(f"{file_path} done!")


        if split_data:
            npz_path = os.path.join(save_folder, os.path.basename(file_path) + ".npz")
            print(f"{npz_path} saved!")
            np.savez(npz_path, pointnormals = sample, points = points, sdfs = sdf)

        return sample, points, sdf, True
    except:
        traceback.print_exc()
        return None, None, None, False


def get_voxels(voxel_model_mat, save_points=False, save_path=None):
    voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
    voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32) - 1
    voxel_model_256 = np.zeros([256, 256, 256], np.uint8)

    for i in range(16):
        for j in range(16):
            for k in range(16):
                voxel_model_256[i * 16:i * 16 + 16, j * 16:j * 16 + 16, k * 16:k * 16 + 16] = voxel_model_b[
                    voxel_model_bi[i, j, k]]
    # add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2, 1, 0)), 2)
    if save_points:
        save_voxels(save_path, voxel_model_256)

    return voxel_model_256


def save_voxels(save_path, voxel_model_256):
    indices = (np.array(np.where(voxel_model_256 > 0.5)) + 0.5) / 256 - 0.5
    np.savetxt(save_path, indices.T)


if __name__ == '__main__':
    debugger = MyDebugger("sample-points-sdf", is_save_print_to_file=False)

    # obj_names_path = '/data/edward/data/all_vox256_img/all_vox256_img_train.txt'
    # obj_names_path = '/data/edward/data_per_category/02691156_airplane/02691156_vox256_img_train.txt'
    # obj_names_path = '/data/edward/data_per_category/02958343_car/02958343_vox256_img_train.txt'
    obj_names_path = r'Y:\data_per_category\03001627_chair\03001627_vox256_img_train.txt'
    # obj_names_path = '/data/edward/data_per_category/chair_test/03001627_vox256_img_train.txt'
    output_path = debugger.file_path(os.path.basename(obj_names_path[:-4] + ".hdf5"))
    # output_path = '/data/edward/data/all_vox256_img/all_vox256_img_train_pointnormals.hdf5'
    mesh_folder = r'/research/dept6/khhui/proj51_backup/edward/ShapeNetCore.v1'
    voxels_folder = r'/research/dept6/khhui/proj51_backup/edward/shapenet/modelBlockedVoxels256/'
    pc_number_of_points = 50000
    sdf_number_of_points = 500000
    # mesh_folder = '/data/edward/Mesh'
    obj_txt = open(obj_names_path, "r")
    obj_list = obj_txt.readlines()
    obj_txt.close()
    obj_list = [item.strip().split('/') for item in obj_list]

    obj_paths = [os.path.join(mesh_folder, class_name, file_name) for class_name, file_name in obj_list]
    voxels_paths = [os.path.join(voxels_folder, class_name, file_name + '.mat') for class_name, file_name in obj_list]
    workers = 15
    indices = None
    split_data = False
    sample_data(file_paths=obj_paths, voxels_paths=voxels_paths, output_file=output_path, normalize=False,
                save_ply=False, save_folder=debugger.file_path('.'),
                pc_number_of_points=pc_number_of_points, sdf_number_of_points = sdf_number_of_points, view_sdf =  False, workers = workers,
                indices = indices, split_data = split_data)
