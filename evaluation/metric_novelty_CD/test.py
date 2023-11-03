from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from pprint import pprint
import pandas as pd

import os
import torch
import numpy as np
import torch.nn as nn
import open3d as o3d


device = torch.device("cuda")

def read_point_clouds(folder, format=".ply"):
    pcds = []
    for i, file in enumerate(os.listdir(folder)):
        if i > 2000:
            continue
        if not file.endswith(format):
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(folder, file))
        pcds.append(np.asarray(pcd.points))
        # print(f"reading file: {file}")
    pcds = np.stack(pcds, axis=0)
    return pcds
    

if __name__ == "__main__":
    num_samples = 2048
    # folder_ref = "/data/ssd/jingyu/metrics/refpoiss_04379243_table_normalize"
    # folder_ref = "/data/ssd/jingyu/metrics/refpoiss_03001627_chair_normalize"
    folder_ref = "/research/dept6/khhui/jyhu_data/refpoiss_02691156_airplane_normalize"
    folder_sample = "/research/dept6/khhui/jyhu_data/sample_GET3D_airplane_normalize"
    print("comparing folder:", folder_ref, folder_sample)

    all_ref = read_point_clouds(folder_ref)

    point_cnt = 2000 #len(all_ref)
    all_sample = read_point_clouds(folder_sample)[:point_cnt]

    print("Finished reading files, staring computing metrics", len(all_ref), "  ", len(all_sample))
    #convert to tensor
    ref_pcs = torch.from_numpy(all_ref).to(device).float()
    sample_pcs = torch.from_numpy(all_sample).to(device).float()
    
    # jsd = JSD(all_sample, all_ref)
    # print("JSD:%s" % jsd)
    
    results = compute_all_metrics(sample_pcs, ref_pcs, batch_size=2000, accelerated_cd=True)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    results = {k:[v] for k, v in results.items()}
    results = pd.DataFrame.from_dict(results)
    results.to_csv(folder_sample.split("/")[-1] + ".csv")
    pprint(results)
    
    
    
    