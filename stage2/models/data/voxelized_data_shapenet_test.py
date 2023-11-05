from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp,random
import trimesh
import torch,glob
import importlib
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian

class VoxelizedDataset(Dataset):


    def __init__(self, mode, res = 32,  voxelized_pointcloud = True, pointcloud_samples = 3000, data_path = '', split_file = '../shapenet/split.npz',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        #self.split = np.load(split_file)

        self.data=[]
        #self.data2 = self.split[mode]
        '''for path in self.data2:
          #print (path)
          name=path.split('/')[-1]#.split('.')[0]
          if '03001627' in path:
            if os.path.exists('/mnt/sda/lzz/ImplicitWavelet-text/voxel_low/'+name+'.npy'):

              self.data.append(path)'''

        #data0=glob.glob('feat/*') 
        #data0=glob.glob('/mnt/sda/lzz/Implicit-manipulate/feat_final3/*')
        #data0=glob.glob('green/*')
        
        #data0=glob.glob('/mnt/sda/lzz/ft-chair-clip-image/feat_evaluation/*')

        #data0=['/mnt/sda/lzz/Implicit-manipulate/feat_final3/a short chair1.npy']
        
        #data0=glob.glob('/mnt/sda/lzz/Implicit-manipulate/feat_final3/with arm*')+glob.glob('/mnt/sda/lzz/Implicit-manipulate/feat_final3/a chair*')


        #data0=glob.glob('/mnt/sda/lzz/ft-chair-gt/feat_evaluation/*')#[10:]
        #data0=glob.glob('feat_final/*')
        data0=glob.glob('../stage1/feat_evaluation/*')
                
        #print (data0,'data0')
        
        #exit()


        



        self.data=[]
        
        for path in data0:
          name=path.split('/')[-1].split('_')[0] #[:-7]
          self.data.append(path)
        
        self.data.sort()
        
        #print (self.data)
        #exit()
        
        
        
        
        
        '''self.data=[]
        
        dic=np.load('../blip_balance100.npy',allow_pickle=1)[()]
        
        
        
        for key in dic.keys():
        

          self.data.append(key)'''
        
        
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



        config_path = 'configs/config.py' #os.path.join(testing_folder, 'config.py')
        
        ## import config here
        spec = importlib.util.spec_from_file_location('*', config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        self.dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=3, wave=config.wavelet, mode=config.padding_mode).cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]
        name=self.data[idx].split('/')[-1].split('_')[0]
        #print (name)

        if 1:# not self.voxelized_pointcloud:




            
            
            
            #path='../jingyudata/03001627_test/'+name+'.npy'
            low_samples=torch.from_numpy(np.load(path)).cuda()
            
            
            #print (low_samples.shape, 'low_samples')
            
            #low_samples=torch.flip(low_samples,[2])
            highs_samples=[torch.zeros((1, 1, 256, 256, 256)).cuda(), torch.zeros((1, 1, 136, 136, 136)).cuda(), torch.zeros((1, 1, 76, 76, 76)).cuda()]
            voxels_pred = self.dwt_inverse_3d_lap((low_samples, highs_samples))
            
            
            voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)

            voxels_pred=voxels_pred.detach().cpu().numpy()
            #voxels_pred=np.flip(np.transpose(voxels_pred, (0,1,4,3,2)),2)
            
            voxels_pred=voxels_pred[:,:,14:-14:2,14:-14:2,14:-14:2]
            input = np.reshape(voxels_pred, (self.res,)*3)



            '''low_samples=torch.from_numpy(np.load('/mnt/sda/lzz/Implicit-manipulate/ablation3/a tall chair.npy')).cuda()
            highs_samples=[torch.zeros((1, 1, 256, 256, 256)).cuda(), torch.zeros((1, 1, 136, 136, 136)).cuda(), torch.zeros((1, 1, 76, 76, 76)).cuda()]
            voxels_pred = self.dwt_inverse_3d_lap((low_samples, highs_samples))
            
            
            voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)
            #voxels_pred=voxels_pred/0.9 

            voxels_pred=voxels_pred.detach().cpu().numpy()
            #voxels_pred=np.flip(np.transpose(voxels_pred, (0,1,4,3,2)),2)
            
            voxels_pred=voxels_pred[:,:,14:-14:2,14:-14:2,14:-14:2]
            input2 = np.reshape(voxels_pred, (self.res,)*3)'''
            
        
        
        return {'inputs': np.array(input, dtype=np.float32), 'path' : path}


        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
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

    def get_loader(self, shuffle =False):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=0,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
