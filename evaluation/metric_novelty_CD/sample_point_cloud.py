import point_cloud_utils as pcu
import os
from trimesh.sample import sample_surface
import trimesh
from multiprocessing import Pool
import random
import open3d as o3d
import traceback
import numpy as np
# saved_folder = "/data/ssd/jingyu/metrics/samplepoisson_imnet_chair/"
# read_folder = "/data/ssd/jingyu/IM-NET/IMGAN/chair_samples/"
# read_folder = "/data/ssd/jingyu/metrics/samplepoisson_ours0"
# saved_folder = "/data/ssd/jingyu/metrics/samplepoisson_ours0"
# saved_folder = "/data/ssd/jingyu/metrics/samplepoisson_imnet_table"
# read_folder = "/data/ssd/jingyu/implicit-decoder/IMGAN/samples/"
# read_folder = "/data/ssd/jingyu/spaghetti/assets/checkpoints/spaghetti_chairs_large/samples/occ/"
# saved_folder = "/data/ssd/jingyu/metrics/samplepoisson_spaghetti_chair"

# read_folder = "/data/ssd/jingyu/3d_gen/shapegan/generated_objects"
# saved_folder = "/data/ssd/jingyu/metrics/samplepoisson_voxel_table"

# read_folder = "/data/ssd/jingyu/metrics/samplepoisson_imnet_chair0/samples/"
# saved_folder = "/data/ssd/jingyu/metrics/samplepoisson_imnet_chair0/"
read_folder = "/mnt/sda/lzz/ifnet-all-text-2branch-batch-feat345-cct-test-3att-highreso-xyz-balance-500-blip-gan/mesh_blip2_5/"
saved_folder = "lzz"

def sample_point_cloud(args, sample_num=2048):
    try:
        path = args[0]
        name = args[1]
        if path.endswith(".dae"):
            dae_dict = trimesh.exchange.dae.load_collada(path)
            # mesh = trimesh.Trimesh(vertices=dae_dict['geometry']['geometry0.0']['vertices'], faces=dae_dict['geometry']['geometry0.0']['faces'])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(dae_dict['geometry']['geometry0.0']['vertices'])
            mesh.triangles = o3d.utility.Vector3iVector(dae_dict['geometry']['geometry0.0']['faces'].astype(np.int32))
            pcd = mesh.sample_points_poisson_disk(sample_num)
            o3d.io.write_point_cloud(os.path.join(saved_folder, name.split(".")[0] + ".ply"), pcd)
        else:
            mesh = o3d.io.read_triangle_mesh(path)
            if "shapegan" in path:
                R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
                mesh.rotate(R, center=(0, 0, 0))
            pcd = mesh.sample_points_poisson_disk(sample_num)
            o3d.io.write_point_cloud(os.path.join(saved_folder, name.split(".")[0] + ".ply"), pcd)
        # cloud = trimesh.PointCloud(mesh.sample(sample_num))
        # cloud.export(os.path.join(saved_folder, name.split(".")[0]+".ply"))
        print(f"Successfully export {name}")
    except:
        traceback.print_exc()
        print(f"Failed export")
    return True
    


if __name__ == "__main__":
    sample_num = 2000
    # mesh format
    format = ".obj"
    num_thread = 20
    paths = os.listdir(read_folder)
    paths = [[os.path.join(read_folder, path), path] for path in paths if path.endswith(format)]
    random.shuffle(paths)
    paths = paths[:sample_num]
    print(saved_folder.split("/")[-1])
    with Pool(num_thread) as p:
        p.map(sample_point_cloud, paths)
    
