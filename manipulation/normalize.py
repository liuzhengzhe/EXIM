import numpy as np
import os
import open3d as o3d
import trimesh
import torch
import argparse


def pc_normalize(pcs): 
    for i in range(pcs.shape[0]):
        pc = pcs[i]
        l = pc.shape[0] 
        centroid = np.mean(pc, axis=0) 
        pc = pc - centroid 
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) 
        pc = pc / m 
        pcs[i] = pc
    return pcs

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
    
def normalize_point_clouds_ours(pcs):
    pcs = 2*(pcs/0.9)
    return pcs
    
def read_point_clouds(folder, format=".ply"):
    pcds = []
    names = []
    for file in os.listdir(folder):
        if not file.endswith(format):
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(folder, file))
        pcds.append(np.asarray(pcd.points))
        names.append(file[:-4])
        # print(f"reading file: {file}")
    pcds = np.stack(pcds, axis=0)
    return pcds, names
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--ours", action='store_true', default=False)
    args = parser.parse_args()

    read_dir = './' #'/mnt/sda/lzz/if-net/experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/evaluation_902_@256/generation/pc/' #'gt_rot' #'/mnt/sda/lzz/if-net/experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/evaluation_269_@256/generation/pc/' #'/mnt/sda/lzz/ImplicitWavelet-generation/pc/' #'gt_rot' #os.path.join("/data/ssd/jingyu/metrics", args.data_path)

    saved_dir = './' #read_dir + "_normalize"
    
    pcds, names=read_point_clouds(read_dir)
    

    norm_pcds = normalize_point_clouds(pcds)
    for i in range(len(norm_pcds)):
        Cloud = trimesh.PointCloud(norm_pcds[i])
        Cloud.export(os.path.join(saved_dir, f"{names[i]}.ply"))
    
    

    
