import os
import torch
import random
import numpy as np
from multiprocessing import Pool

def convert_file(args):

    idx, path = args
    assert path.endswith('.npy')
    voxels_np = np.load(path)
    save_path = path[:-4] + '.pt'

    if not os.path.exists(save_path):
        voxels_torch = torch.from_numpy(voxels_np).float()
        torch.save(voxels_torch, save_path)
    print(f"{idx} : {save_path} Done!")


if __name__ == '__main__':
    save_folder = r'Y:\sdf_samples\03001627'
    workers = 6

    paths = [ os.path.join(save_folder, file) for file in os.listdir(save_folder) if file.endswith('.npy') and not os.path.exists(os.path.join(save_folder,
                                                                                                           file[:-4] + '.pt'))]

    random.shuffle(paths)

    args = [ (idx, path) for idx, path in enumerate(paths) ]
    print(f"{len(args)} left to be processed!")

    pool = Pool(workers)
    pool.map(convert_file, args)
