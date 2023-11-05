from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch,glob



class VoxelizedDataset(Dataset):


    def __init__(self, mode, res = 32,  voxelized_pointcloud = True, pointcloud_samples = 3000, data_path = '../shapenet/data/', split_file = '../shapenet/split.npz',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        #self.split = np.load(split_file)

        #self.data= glob.glob('/mnt/sda/lzz/ImplicitWavelet-text/voxel_low/*.npy') #[]
        self.data= []
        self.data2 = glob.glob('../data/03001627_train/*.npy') #'/mnt/sda/lzz/ImplicitWavelet-text/voxel_low/*.npy') #self.split[mode]
        
        
        for path in self.data2:
          name=path.split('/')[-1]
          if os.path.exists('../data/shapenet/data/03001627/'+name[:-4]): #os.path.exists('../ifnet/shapenet/data/03001627/'+name[:-4]):

            self.data.append(path)


        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



    def __len__(self):
        return len(self.data)
        
        

    def __getitem__(self, idx):
        path = self.path + self.data[idx]
        name=self.data[idx].split('/')[-1]
        #print (name)

        if 1:# not self.voxelized_pointcloud:

            input = np.load('../data/03001627_train/'+name) #path + '/voxelization_{}.npy'.format(self.res))
            #print ('name',name,flush=True)
            #exit()

            '''highs_samples=[torch.zeros((1, 1, 256, 256, 256)).cuda(), torch.zeros((1, 1, 136, 136, 136)).cuda(), torch.zeros((1, 1, 76, 76, 76)).cuda()]
            voxels_pred = self.dwt_inverse_3d_lap((low_samples, highs_samples))
            
            
            voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)
            voxels_pred/=0.9

            voxels_pred=voxels_pred.detach().cpu().numpy()
            voxels_pred=np.flip(np.transpose(voxels_pred, (0,1,4,3,2)),2)
            
            voxels_pred=voxels_pred[:,:,14:-14:2,14:-14:2,14:-14:2]


      
            occupancies = np.reshape(voxels_pred, (self.res,)*3)'''
         


            '''occupancies = np.load(path + '/voxelization_{}.npy'.format(self.res))
            occupancies = np.unpackbits(occupancies)
            input = np.reshape(occupancies, (self.res,)*3)'''
        '''else:
            voxel_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
            #voxel_path='/mnt/sda/lzz/ImplicitWavelet-text/voxel_low/'+name+'.npy'

            occupancies = np.unpackbits(np.load(voxel_path)) #['compressed_occupancies'])

            input = np.reshape(occupancies, (self.res,)*3)'''

        #print (input.shape, 'input', np.unique(input))
        #exit()

        '''voxels_pred=np.load('/mnt/sda/lzz/ImplicitWavelet-text/1.npy')
        print (np.unique(voxels_pred))
        #print (input.shape, data.shape, 'dat')
        #input[:]=data[0,0,::2,::2,::2]
        
        voxels_pred=torch.from_numpy(voxels_pred).cuda()
        voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)
        voxels_pred/=0.9
        
        voxels_pred=voxels_pred.detach().cpu().numpy()
        voxels_pred=np.flip(np.transpose(voxels_pred, (0,1,4,3,2)),2)
        
        voxels_pred=voxels_pred[:,:,14:-14:2,14:-14:2,14:-14:2]
        
        voxels_pred2=voxels_pred.copy()
        voxels_pred2[np.where(voxels_pred>0)]=0
        voxels_pred2[np.where(voxels_pred<0)]=1'''
        
        
        '''voxels_pred=np.load('/mnt/sda/lzz/ImplicitWavelet-text/voxel_high/3d629d27b74fad91dbbc9440457e303e.npy') #[:,:,::2,::2,::2]
        voxels_pred2=voxels_pred.copy()
        voxels_pred2[np.where(voxels_pred>0)]=0
        voxels_pred2[np.where(voxels_pred<0)]=1
        input[:]=voxels_pred2#[0,0,::2,::2,::2]'''
        
        
        
        #return {'occupancies': np.array(occupancies, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}


        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = '../data/shapenet/data/03001627/'+name[:-4]+'/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
            #print (boundary_samples_path)
            boundary_samples_npz = np.load(boundary_samples_path,allow_pickle=True)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])


        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        return {'grid_coords':np.array(coords, dtype=np.float32),'occupancies': np.array(occupancies, dtype=np.float32),'points':np.array(points, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'path' : path}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
