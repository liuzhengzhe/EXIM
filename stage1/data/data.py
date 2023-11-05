import torch
import os,random
import numpy as np
from tqdm import tqdm
from models.network import create_coordinates
import random
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SDFSamples(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 data_folder : str,
                 resolution: int,
                 num_points: int,
                 use_surface_samples : False,
                 sample_resolution = 64,
                 sample_ratio = 0.5,
                 interval: int = 1,
                 load_ram = False,
                 use_preload = False,
                 data_files=None,
                 first_k = None):
        super(SDFSamples, self).__init__()

        ### get file
        if data_files is None:
            data_files = []

        ## load the data folder
        self.use_preload = use_preload
        #print (',self.use_preload',self.use_preload)
        self.data_preloaded = [] #[ np.load(data_file[0]) for data_file in data_files ] if self.use_preload else []

        label_txt_path = data_path
        import glob
        self.data_path0  = glob.glob('../data/03001627_train/*.npy') #+glob.glob('../3DFront/4/*.npy')
        self.data_path0.sort()
        
        self.data_path=[]
        
        
        filters=[]
        f=open('filter.txt')
        for line in f:
          line=line.strip()
          filters.append(line)
        
        for data_path in self.data_path0:
          name=data_path.split('/')[-1].split('.')[0]
          if name in filters:
            continue
          else:
            self.data_path.append(data_path)
        
        
        #self.data_path=self.data_path 

        ## interval
        self.interval = interval
        self.resolution = resolution
        self.num_points = num_points

        ## indices
        self.all_indices_without_permute = create_coordinates(resolution).view((resolution, resolution, resolution, 3)).int().cpu().numpy()
        all_indices = create_coordinates(resolution).view((-1, 3))
        all_indices = all_indices[torch.randperm(all_indices.size(0))]
        self.all_indices = all_indices.view((-1, num_points, 3)).int().cpu().numpy()

        ### use coefficient
        self.return_all = False

        ### samples
        self.sample_ratio = sample_ratio
        self.use_surface_samples = use_surface_samples
        self.sample_resolution = sample_resolution
        self.sample_voxel_size = resolution // self.sample_resolution
        self.sample_coordinates = create_coordinates(sample_resolution).int().cpu().numpy().reshape((-1, 3))


        ### sample
        self.sample_index = None

        ### data length
        self.data_len = len(self.data_path) #// self.interval if len(self.data_preloaded) == 0 else self.data_preloaded[0].shape[0]
        self.dic=np.load('../data/official_chair_train.npy',allow_pickle=1)[()]
        
        
        
        
        #self.dic_front=np.load('../3dfront_dic.npy',allow_pickle=True)[()]
            


    def __len__(self):
        #print (self.data_len)

        return self.data_len

    def __getitem__(self, idx):

        idx = idx * self.interval
        
        source=self.data_path[idx].split('/')[-3]
        
        category=self.data_path[idx].split('/')[-2]
        name=self.data_path[idx].split('/')[-1].split('.')[0]
        data3=np.load(self.data_path[idx])#[:23,:23,:23]
        
        
        
        #data2=np.load(self.data_path[idx].replace('data3','data2'))
        
        mode=0
        
        clip_feature=np.zeros((1,768))
        
        #print ('source', source, name, flush=True)
        if source=='3DFront':
          if random.random()<0.0:
            text='      '
            
            clip_feature=np.load('../clip_feat/'+name+'.npy')
            mode=1
            
          else:
            text_idx=random.randint(0,len(self.dic_front[name])-1)
            text=self.dic_front[name][text_idx]
            
            text=' '.join(text.split(' ')[:20])          
          
          
        else:
  
          text_idx=random.randint(0,len(self.dic[name])-1)
          text=self.dic[name][text_idx]
          
          text=' '.join(text.split(' ')[4:20])
  

        processed_data=(data3, data3)

        

        '''if self.use_preload:
            processed_data = tuple([data[idx] for data in self.data_preloaded])
        else:
            data_path = self.data_path[idx]
            if self.load_ram:
                voxels_sdf = self.loaded_data[idx]
            else:
                voxels_sdf = np.load(data_path)

            if self.sample_index is None:
                batch_idx = np.random.choice(self.all_indices.shape[0])
                indices = self.all_indices[batch_idx]
            else:
                indices = self.all_indices[self.sample_index]

            if self.use_surface_samples:
                ### compute samples
                data_sample_path = data_path[:-4] + f'_{self.sample_resolution}.npz'
                if os.path.exists(data_sample_path):
                    data_samples = np.load(data_sample_path, allow_pickle = True)
                    sign_changed_voxel_indices = data_samples['sign_changed_voxel_indices']
                    sign_unchanged_voxel_indices = data_samples['sign_unchanged_voxel_indices']
                else:
                    sampled_voxels_values = np.concatenate([voxels_sdf[self.sample_coordinates[i, 0] * self.sample_voxel_size:
                                                                       self.sample_coordinates[i, 0] * self.sample_voxel_size + self.sample_voxel_size,
                                                                       self.sample_coordinates[i, 1] * self.sample_voxel_size:
                                                                       self.sample_coordinates[i, 1] * self.sample_voxel_size + self.sample_voxel_size,
                                                                       self.sample_coordinates[i, 2] * self.sample_voxel_size:
                                                                       self.sample_coordinates[i, 2] * self.sample_voxel_size + self.sample_voxel_size][None, :, :, :] for i in
                                                                       range(self.sample_coordinates.shape[0])], axis=0)
                    sampled_voxels_indices = np.concatenate([self.all_indices_without_permute[self.sample_coordinates[i, 0] * self.sample_voxel_size:
                                                                       self.sample_coordinates[i, 0] * self.sample_voxel_size + self.sample_voxel_size,
                                                                       self.sample_coordinates[i, 1] * self.sample_voxel_size:
                                                                       self.sample_coordinates[i, 1] * self.sample_voxel_size + self.sample_voxel_size,
                                                                       self.sample_coordinates[i, 2] * self.sample_voxel_size:
                                                                       self.sample_coordinates[i, 2] * self.sample_voxel_size + self.sample_voxel_size][None, :, :, :] for i in
                                                                       range(self.sample_coordinates.shape[0])], axis=0)
                    samples_voxels_sign_changed = np.sign(np.max(sampled_voxels_values, axis=(1, 2, 3))) != np.sign(
                        np.min(sampled_voxels_values, axis=(1, 2, 3)))
                    sign_changed_voxel_indices = sampled_voxels_indices[samples_voxels_sign_changed].reshape((-1, 3))
                    sign_unchanged_voxel_indices = sampled_voxels_indices[np.logical_not(samples_voxels_sign_changed)].reshape((-1, 3))
                    file = open(data_sample_path, 'wb')
                    np.savez(file, sign_changed_voxel_indices=sign_changed_voxel_indices, sign_unchanged_voxel_indices=sign_unchanged_voxel_indices)
                    data_samples = np.load(data_sample_path, allow_pickle = True)
                    sign_changed_voxel_indices = data_samples['sign_changed_voxel_indices']
                    sign_unchanged_voxel_indices = data_samples['sign_unchanged_voxel_indices']

                sampled_surface_indices = np.random.randint(0, sign_changed_voxel_indices.shape[0], int(self.num_points * self.sample_ratio))
                sampled_off_surface_indices = np.random.randint(0, sign_unchanged_vo
                
                
                
                
                
                
                xel_indices.shape[0], self.num_points - int(self.num_points * self.sample_ratio))

                signed_changed_samples = np.take(sign_changed_voxel_indices, sampled_surface_indices, axis = 0)
                signed_unchanged_samples = np.take(sign_unchanged_voxel_indices, sampled_off_surface_indices, axis = 0)

                indices = np.concatenate((signed_changed_samples, signed_unchanged_samples), axis = 0)

            voxels_sdf_gt = voxels_sdf[indices[:, 0], indices[:, 1], indices[:, 2]]

            if self.return_all:
                processed_data = (indices, voxels_sdf_gt, voxels_sdf)
            else:
                processed_data = (indices, voxels_sdf_gt)'''

        #print (type(text), type(idx), type(clip_feature), type(mode))
        return processed_data, text, idx, clip_feature, mode
