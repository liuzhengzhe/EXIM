from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from metrics.evaluation_metrics import _pairwise_EMD_CD_
from pprint import pprint
import pandas as pd
import traceback 
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import open3d as o3d
import pandas as pd
import glob

device = torch.device("cuda")

def sample_point_cloud(path, sample_num=2048):
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_poisson_disk(sample_num)
        pcd = np.array(pcd.points)
        print(pcd.shape)
        return pcd, True
    except:
        return None, False


def normalize_point_clouds(pc):
    pc = torch.from_numpy(pc)
    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
    shift = ((pc_min + pc_max) / 2).view(1, 3)
    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
    pc = (pc - shift) / scale
    pc = pc.numpy()
    return pc

def read_point_clouds(folder, format=".ply"):
    pcds, names = [], []

    for i, file in tqdm(list(enumerate(os.listdir(folder)))):
        if not file.endswith(format):
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(folder, file))
        pcds.append(np.asarray(pcd.points))
        # print(f"reading file: {file}")
        names.append(os.path.basename(file))
        # print(f"loaded {file}")
    pcds = np.stack(pcds, axis=0)
    return pcds, names
    

if __name__ == "__main__":
    num_samples = 2048
    folder_sample = "lzz_normalize/"
    shapenet_folder = '/mnt/sda/lzz/metric_jingyu/gt2k/' #'/mnt/sda/lzz/metric_jingyu/gtpc/' #/research/dept6/khhui/proj51_backup/edward/ShapeNetCore.v1'
    #training_text = '/research/dept6/khhui/data_per_category/03001627_chair/03001627_vox256_img_train.txt' # from iment
    #filename = 'model_flipped_manifold.obj'

    gen_pts, names = read_point_clouds(folder_sample)


    '''obj_txt = open(training_text, "r")
    obj_list = obj_txt.readlines()
    obj_txt.close()
    obj_list = [item.strip().split('/') for item in obj_list][:5]'''

    # processing sampling
    point_clouds = []
    '''for cat, md5 in tqdm(list(obj_list)):
        mesh_path = os.path.join(shapenet_folder, cat, md5, filename)
        pts_cloud, valid = sample_point_cloud(mesh_path)
        if valid:
            pts_cloud = normalize_point_clouds(pts_cloud)
            point_clouds.append(pts_cloud[None, :])'''
    
    for path in glob.glob(shapenet_folder+'/*.ply'):
      pcd = o3d.io.read_point_cloud(path)
      point_clouds.append(np.asarray(pcd.points))
    
    

    train_pts = np.concatenate(point_clouds, axis = 0)

    train_pts = torch.from_numpy(train_pts).to(device).float()
    gen_pts = torch.from_numpy(gen_pts).to(device).float()

    cd_results, emd_results = _pairwise_EMD_CD_(gen_pts, train_pts, batch_size=500, accelerated_cd=True)
    print(cd_results.size(), emd_results.size())

    cd_dist = np.min(cd_results.cpu().numpy(), axis = 1)
    emd_dist = np.min(cd_results.cpu().numpy(), axis = 1)  

    pd.DataFrame.from_dict({'name' : names, 'dist' : cd_dist}).to_csv('./cd_dist.csv')
    pd.DataFrame.from_dict({'name' : names, 'dist' : cd_dist}).to_csv('./emd_dist.csv')



    # print(train_pts.size(), gen_pts.size())
        # print(f"Loading {mesh_path}")