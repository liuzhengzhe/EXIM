import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
from glob import glob
from Common import point_operation
import os
warnings.filterwarnings('ignore')
from torchvision import transforms
from Common import data_utils as d_utils
from Common import point_operation
import torch

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['poisson_2048'][:]
    return data


class H5DataLoader(Dataset):
    def __init__(self, opts,augment=False, partition='train'):
        self.opts = opts
        h5_file = os.path.join(opts.data_root, str(opts.choice).lower()+".h5")
        print("---------------h5_file:",h5_file)
        self.data = load_h5(h5_file)
        self.data = self.opts.scale * point_operation.normalize_point_cloud(self.data)
        self.num_points = opts.np
        self.augment = augment
        self.partition = partition

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point_set = self.data[index][:self.num_points,:3].copy()
        np.random.shuffle(point_set)

        if self.augment:
            point_set = point_operation.rotate_point_cloud_and_gt(point_set)
            point_set = point_operation.random_scale_point_cloud_and_gt(point_set)
        point_set = point_set.astype(np.float32)

        return torch.from_numpy(point_set)
