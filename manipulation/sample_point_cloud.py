import point_cloud_utils as pcu
import os
from trimesh.sample import sample_surface
import trimesh
from multiprocessing import Pool
import random
import open3d as o3d
import traceback, glob
import numpy as np



def denoise(mesh):
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    #print (cluster_n_triangles)

    cluster_area = np.asarray(cluster_area)

    largest_cluster_idx = cluster_n_triangles.argmax()
    
    idxs=np.where(cluster_n_triangles<10000)[0]
    
    #print (largest_cluster_idx,'largest_cluster_idx')
    triangles_to_remove = triangle_clusters != largest_cluster_idx #& cluster_n_triangles<10000
    triangles_to_remove[:]=False
    #print (triangles_to_remove.shape,triangle_clusters.shape,triangle_clusters, np.unique(triangle_clusters))
    for idx in idxs:
      #print ('idx',idx, np.where(triangle_clusters==idx))
      triangles_to_remove[np.where(triangle_clusters==idx)]=True
    #print (np.unique(triangles_to_remove), 'triangles_to_remove')
    mesh.remove_triangles_by_mask(triangles_to_remove)
    
    return mesh
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
# read_folder = "/data/ssd/jingyu/metrics/0_selected_query_chairs"
# saved_folder = "/data/ssd/jingyu/metrics/0_selected_query_chairs"
#read_folder = "/mnt/sda/lzz/if-net/experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/evaluation_902_@256/generation/off/"
#saved_folder = "/mnt/sda/lzz/if-net/experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/evaluation_902_@256/generation/pc/"
#read_folder='/mnt/sda/lzz/ImplicitWavelet-generation/normalize/'
#saved_folder = '/mnt/sda/lzz/ImplicitWavelet-generation/pc/'

#read_folder='/mnt/sda/lzz/Wavelet-Generation/debug/2023-04-07_15-03-19_Network-Marching-Cubes-Diffusion-Gen/'
#saved_folder = 'uncond_pc_edward'
read_folder='./' #'/home/zzliu/generated_chair_meshes/' #'/mnt/sda/lzz/ifnet/mesh/'
saved_folder = './'
#read_folder='/mnt/sda/lzz/jingyudata/generated_chair_meshes/'#'/mnt/sda/lzz/Wavelet-Generation/debug/2023-04-06_16-41-17_Network-Marching-Cubes-Diffusion-Gen/'
#saved_folder = 'uncond_pc_our_r'

def normalize_point_clouds(pcs):
    pcs = torch.from_numpy(pcs)
    for i in range(pcs.shape[0]):
        pc = pcs[i]
        pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
        pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
        shift = ((pc_min + pc_max) / 2).view(1, 3)
        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    pcs = pcs.numpy()
    return pcs

def sample_point_cloud(path, sample_num=2048):
    try:
    
        #path = args[0]
        #name = args[1]
        name=path.split('/')[-1]
        
        #if os.path.exists(os.path.join(saved_folder, name + ".ply")):
         #   return 1
        
        if path.endswith(".dae"):
            dae_dict = trimesh.exchange.dae.load_collada(path)
            # mesh = trimesh.Trimesh(vertices=dae_dict['geometry']['geometry0.0']['vertices'], faces=dae_dict['geometry']['geometry0.0']['faces'])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(dae_dict['geometry']['geometry0.0']['vertices'])
            mesh.triangles = o3d.utility.Vector3iVector(dae_dict['geometry']['geometry0.0']['faces'].astype(np.int32))
            pcd = mesh.sample_points_poisson_disk(sample_num)
            o3d.io.write_point_cloud(os.path.join(saved_folder, name[:-4] + ".ply"), pcd)
        else:
            mesh = o3d.io.read_triangle_mesh(path)
            
            #mesh=denoise(mesh)
            
            #o3d.io.write_triangle_mesh('denoised/'+name, mesh)

            if "shapegan" in path:
                R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
                mesh.rotate(R, center=(0, 0, 0))
            if "styleSDF" in path:
                R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
                mesh.rotate(R, center=(0, 0, 0))
            if "GET3D" in path:
                R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
                mesh.rotate(R, center=(0, 0, 0))
            pcd = mesh.sample_points_poisson_disk(sample_num)
            #print (pcd.shape)
            o3d.io.write_point_cloud(os.path.join(saved_folder, name + ".ply"), pcd)
        # cloud = trimesh.PointCloud(mesh.sample(sample_num))
        # cloud.export(os.path.join(saved_folder, name.split(".")[0]+".ply"))
        print(f"Successfully export {name}")
    except:
    #    traceback.print_exc()
        print(f"Failed export") 
    return True
    


if __name__ == "__main__":
    sample_num = 2000
    # mesh format
    #format = ".obj"
    num_thread = 40
    #paths = os.listdir(read_folder)
    #paths = [[os.path.join(read_folder, path), path] for path in paths if path.endswith(format)]
    paths=glob.glob('../data/input.obj')
    random.shuffle(paths)
    #paths = paths[:sample_num]
    #print(saved_folder.split("/")[-1])
    #paths.sort()
    #for path in paths:
    #  sample_point_cloud(path)
    with Pool(num_thread) as p:
        p.map(sample_point_cloud, paths)
    
