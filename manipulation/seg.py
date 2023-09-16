import os
import importlib
import torch
import numpy as np
import mcubes
import glob
import torch.nn.functional as F
#from data.data import SDFSamples
import random
from models.network import MultiScaleMLP,SparseComposer, create_coordinates
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, SpacedDiffusion, space_timesteps
from utils.debugger import MyDebugger
testing_folder='../data/'
data_len = 5421 # chair
# data_len = 6807 # table
# data_len = 1256 # cabinet
# data_len = 3234 # airplane
from modelAE import IM_AE
data_path = r'/home/edward/data/03001627_chair/03001627_vox256_img_train.txt'
data_folder = r'Y:\sdf_samples_scaled_0.98\03001627'
from plyfile import PlyData,PlyElement
config_path = os.path.join(testing_folder, 'config.py')

## import config here
spec = importlib.util.spec_from_file_location('*', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
device = torch.device(0)
### debugger
from configs import config as current_config

dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
    device)
dwt_forward_3d_lap = DWTForward3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
    device)

composer_parms = dwt_inverse_3d_lap if config.use_dense_conv else None
dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution],
                                     J=config.max_depth,
                                     wave=config.wavelet, mode=config.padding_mode,
                                     inverse_dwt_module=composer_parms).to(
    device)
        
paths=glob.glob('../data/03001627_train/*')


im_ae = IM_AE()         
model_dir = '../data/IM-NET-pytorch.pth' #fin.readline().strip()
im_ae.im_network.load_state_dict(torch.load(model_dir), strict=True)


im_ae.im_network.eval()

highs_samples = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[i]), device=device) for i in
                                 range(config.max_depth)]

from modelAE import IM_AE

for path in paths:

    print (path)
    name=path.split('/')[-1].split('.')[0]

    data3np=np.load(path) 
    data3=torch.from_numpy(data3np).cuda().unsqueeze(0).unsqueeze(0)
    
    data3=data3



    voxels_input = dwt_inverse_3d_lap((data3, highs_samples))


    batch_voxels = voxels_input.clone() #.astype(np.float32)
    batch_voxels[torch.where(voxels_input<=0)]=1
    batch_voxels[torch.where(voxels_input>0)]=0

    model_z,_ = im_ae.im_network(batch_voxels[:,:,2:-1:4,2:-1:4,2:-1:4], None, None, is_training=False)
    model_float = im_ae.z2voxel(model_z)[1:-1,1:-1,1:-1]
    model_float=torch.nn.functional.interpolate(model_float.float().unsqueeze(0).unsqueeze(0), size=46, mode='nearest').long()[0,0,:,:,:]
    
    #print (diffss_scale.shape, model_float.shape, batch_voxels_down_46.shape, torch.unique(model_float))

    batch_voxels_down=batch_voxels[:,:,2:-1:4,2:-1:4,2:-1:4]
    batch_voxels_down_46=torch.nn.functional.interpolate(batch_voxels, size=46, mode='nearest')
    
    #model_float_small=torch.nn.functional.interpolate(model_float.float().unsqueeze(0).unsqueeze(0), size=36, mode='nearest').long()[0,0,:,:,:]
    #model_float[:]=-1
    #model_float[5:-5,5:-5,5:-5]=model_float_small
    
    batch_voxels_down_46_small=torch.nn.functional.interpolate(batch_voxels_down_46.float(), size=36, mode='nearest').long()#[0,0,:,:,:]
    batch_voxels_down_46[:]=0
    batch_voxels_down_46[:,:,5:-5,5:-5,5:-5]=batch_voxels_down_46_small
    
    np.save('../part/'+name+'.npy', batch_voxels_down_46.detach().cpu().numpy()[0,0])
    

    '''some_array=[]
    for i in range(0,46):
        for j in range(0,46):
            for k in range(0,46):
                if batch_voxels_down_46[0,0,i,j,k]>0.1:
                    some_array.append((i/46-0.5,j/46-0.5,k/46-0.5,model_float[i,j,k]*50,model_float[i,j,k]*50,model_float[i,j,k]*50))
    some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
    el = PlyElement.describe(some_array, 'vertex')
    PlyData([el]).write('part/'+name+'.ply')'''