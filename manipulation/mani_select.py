import os
import importlib
import torch
import numpy as np
import mcubes
import glob
import torch.nn.functional as F
from data.data import SDFSamples
import random
from models.network import MultiScaleMLP,SparseComposer, create_coordinates
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, SpacedDiffusion, space_timesteps
from utils.debugger import MyDebugger
from models.module.diffusion_network import UNetModel, MyUNetModel
import time
from plyfile import PlyData,PlyElement
from diffusers_mani import UNet3DConditionModel as UNet3DConditionModel_mani
from diffusers_ori import UNet3DConditionModel as UNet3DConditionModel_ori

from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from modelAE import IM_AE

def process_state_dict(network_state_dict):
    for key, item in list(network_state_dict.items()):
        if 'module.' in key:
            new_key = key.replace('module.', '')
            network_state_dict[new_key] = item
            del network_state_dict[key]

    return network_state_dict




### Setting
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-03-31_19-42-28_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-06_17-33-46_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-08_13-20-54_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-11_18-07-01_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-13_14-01-48_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-19_11-24-26_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-22_19-45-58_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_00-07-02_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_00-22-05_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_12-16-46_Wavelet-Training-experiment'
# testing_folder = r'E:\proj51_debug\2022-04-26_15-11-28_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_15-17-20_Wavelet-Training-experiment'
# testing_folder = r'E:\proj51_debug\2022-04-26_20-53-49_Wavelet-Training-experiment'
# testing_folder = r'E:\proj51_debug\2022-04-26_21-02-43_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_20-54-07_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_20-55-17_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-27_12-25-56_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-27_21-25-48_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-28_17-46-15_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-27_21-25-48_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-29_13-38-55_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-29_19-47-03_Wavelet-Training-experiment'
# testing_folder = r'E:\testing_folder\exp_6' # Chair
# testing_folder = r'E:\testing_folder\exp_7'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-04_13-59-26_Wavelet-Training-experiment' # Chair
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-07_17-40-22_Wavelet-Training-experiment' # Chair 2
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-10_01-52-15_Wavelet-Training-experiment' # chair 3
# testing_folder = r'E:\testing_folder\2022-05-10_14-12-39_Wavelet-Training-experiment' # chair new batch 16
# testing_folder = r'E:\testing_folder\2022-05-11_16-43-53_Wavelet-Training-experiment' # chair new batch 64
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-12_15-20-17_Wavelet-Training-experiment'
# testing_folder = r'E:\testing_folder\2022-05-13_01-53-53_Wavelet-Training-experiment' # Chair
#testing_folder = 'debug/2022-11-07_12-26-09_Wavelet-Training-experiment'
#testing_folder = 'debug/2022-11-07_12-26-09_Wavelet-Training-experiment/'
#testing_folder = '/mnt/sda/lzz/ImplicitWavelet-text-from0-global/debug/2022-11-20_23-10-19_Wavelet-Training-experiment/'
#testing_folder='/mnt/sda/lzz/ImplicitWavelet-text-from0-global/debug/2022-11-23_23-47-33_Wavelet-Training-experiment/'
#testing_folder='/mnt/sda/lzz/ImplicitWavelet-text-from0/debug/2022-11-23_23-47-51_Wavelet-Training-experiment/'
#testing_folder='/mnt/sda/lzz/ImplicitWavelet-text-from0/debug/2022-11-20_23-11-21_Wavelet-Training-experiment/'
#testing_folder='/mnt/sda/lzz/ImplicitWavelet-text-from0-global/debug/2022-11-23_23-47-33_Wavelet-Training-experiment/'
#testing_folder='/mnt/sda/lzz/ImplicitWavelet-text-mask/debug/2022-12-18_17-51-13_Wavelet-Training-experiment/'
testing_folder='/mnt/sda/lzz/ImplicitWavelet-generation/debug/2022-11-23_23-47-33_Wavelet-Training-experiment/'
data_len = 5421 # chair
# data_len = 6807 # table
# data_len = 1256 # cabinet
# data_len = 3234 # airplane

data_path = r'/home/edward/data/03001627_chair/03001627_vox256_img_train.txt'
data_folder = r'Y:\sdf_samples_scaled_0.98\03001627'

config_path = os.path.join(testing_folder, 'config.py')

## import config here
spec = importlib.util.spec_from_file_location('*', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

### debugger
from configs import config as current_config
model_type = f"Wavelet-Decoding"


#four legs 600, 0.3
#no armrest 600, 0.5
#tall back 

# epoch = 145 # table
# epoch = 500 # OLD CHAIR
# epoch = 450 # CHAIR
# epoch = 600 #225 #165 # chair 2
epoch = 2000 # chair 3
# epoch = 620 # table
# epoch = 5000
# epoch = 4080 # airplane
stage = 3
config.batch_num_points = 32 ** 3
use_preload = False
#loading_files = [(r'E:\03001627_3.npy', 3)]
# loading_files = [(r'E:\03001627_3.npy', 3), (r'E:\03001627_2.npy', 2)]
loading_files = [(r'E:\03001627_0.1_bior6.8_3.npy', 3), (r'E:\03001627_0.1_bior6.8_2.npy', 2)]

###
testing_cnt = 24
extra_start = 0
test_index = config.training_stage
clip_noise = False
evaluate_gt = False
fixed_noise = False

noise_path = 'debug/2022-11-21_10-26-45_Network-Marching-Cubes-Diffusion-Gen//4_265_noise.pt' #

need_gt = False
use_ddim = True
save_no_high = True
ddim_eta = 1.0
respacing = [config.diffusion_step // 10]
#noise_path = r'E:\proj51_debug\2022-05-17_12-42-13_Network-Marching-Cubes-Diffusion-Gen\229_265_noise.pt' #

#
use_high_level_network = False


import os


im_ae = IM_AE()         
model_dir = '../data/IM-NET-pytorch.pth' #fin.readline().strip()
im_ae.im_network.load_state_dict(torch.load(model_dir), strict=True)


im_ae.im_network.eval()

def dilate(img,ksize=5):
  p=(ksize-1)//2
  img=F.pad(img, pad=[p,p,p,p,p,p], mode='reflect')
  out=F.max_pool3d(img, kernel_size=ksize, stride=1, padding=0)
  return out

def erode(img, ksize=3):
  out=1-dilate(1-img, ksize)
  return out
  
  
def write_ply_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def one_generation_process(args):
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    cuda_id, start_index, testing_cnt, folder_path = args
    device = torch.device(f'cuda:{cuda_id}')
    network_path = os.path.join(testing_folder, f'model_epoch_{stage}_{epoch}.pth')
    network_path2 = '../data/manipulate_epoch_3_1300.pth'
    print ('network', network_path)
    ### create dataset
    samples = SDFSamples(data_path=data_path,
                         data_folder=data_folder,
                         resolution=config.resolution,
                         num_points=config.num_points,
                         interval=config.interval,
                         first_k=config.first_k,
                         use_surface_samples=False,
                         data_files=loading_files,
                         use_preload=use_preload
                         )
    ### level_indices_remap
    level_map = {idx: level for idx, (_, level) in enumerate(loading_files)}

    samples.return_all = True
    with torch.no_grad():
        ### initialize network
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
        
        '''model_path_clip = "openai/clip-vit-large-patch14"
        clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
        clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float32)
        clip = clip_model.text_model.float().to(device)'''

        network = UNet3DConditionModel_ori() 


        network_state_dict = torch.load(network_path,map_location=f'cuda:{cuda_id}')
        network_state_dict = process_state_dict(network_state_dict)

        network.load_state_dict(network_state_dict)
        network = network.to(device)
        network.eval()



        network2 = UNet3DConditionModel_mani() 


        network_state_dict = torch.load(network_path2,map_location=f'cuda:{cuda_id}')
        network_state_dict = process_state_dict(network_state_dict)

        network2.load_state_dict(network_state_dict)
        network2 = network2.to(device)
        network2.eval()

        if use_high_level_network:
            high_level_network = MyUNetModel(in_channels= 1,
                                spatial_size= dwt_sparse_composer.shape_list[high_level_config.training_stage][0],
                                model_channels=high_level_config.unet_model_channels,
                                out_channels= 1,
                                num_res_blocks=high_level_config.unet_num_res_blocks,
                                channel_mult=high_level_config.unet_channel_mult,
                                attention_resolutions=high_level_config.attention_resolutions,
                                dropout=0,
                                dims=3)
            high_level_network_state_dict = torch.load(high_level_network_path, map_location=f'cuda:{cuda_id}')
            high_level_network_state_dict = process_state_dict(high_level_network_state_dict)
            high_level_network.load_state_dict(high_level_network_state_dict)
            high_level_network = high_level_network.to(device)
            high_level_network.eval()
        else:
            high_level_network = None


        betas = get_named_beta_schedule(config.diffusion_beta_schedule, config.diffusion_step,
                                        config.diffusion_scale_ratio)

        diffusion_module = SpacedDiffusion(use_timesteps=space_timesteps(config.diffusion_step, respacing),
                                           betas=betas,
                                           model_var_type=config.diffusion_model_var_type,
                                           model_mean_type=config.diffusion_model_mean_type,
                                           loss_type=config.diffusion_loss_type)

        diffusion_module_train = GaussianDiffusion(betas=betas,
                                      model_var_type=config.diffusion_model_var_type,
                                      model_mean_type=config.diffusion_model_mean_type,
                                      loss_type=config.diffusion_loss_type,
                                      rescale_timesteps=config.diffusion_rescale_timestep if hasattr(config, 'diffusion_rescale_timestep') else False)
        testing_indices = [265] * testing_cnt
        if fixed_noise:
            if noise_path is not None:
                noise = torch.load(noise_path, map_location=f'cuda:{cuda_id}').to(device)
            else:
                noise = torch.randn([1, 1] + dwt_sparse_composer.shape_list[test_index]).to(device)
        else:
            noise = None


        #try:
        paths=glob.glob('origin*.npy')+glob.glob('mani*.npy')
        for path in paths:
          os.remove(path)


        for m in range(1):
            testing_sample_index = testing_indices[m]
            if use_preload:
                data = samples[testing_sample_index][0]
                low_lap, highs_lap = None, [None] * config.max_depth
                coeff_gt = data
                for j, gt in enumerate(coeff_gt):
                    level = level_map[j]
                    if level == config.max_depth:
                        low_lap = torch.from_numpy(coeff_gt[j]).unsqueeze(0).unsqueeze(1).to(device)
                    else:
                        highs_lap[level] = torch.from_numpy(coeff_gt[j]).unsqueeze(0).unsqueeze(1).to(device)
            else:
                # voxels_np = np.load(samples.data_path[testing_sample_index])
                # voxels_cuda = torch.from_numpy(voxels_np).float().to(device).unsqueeze(0).unsqueeze(0)
                # low_lap, highs_lap = dwt_forward_3d_lap(voxels_cuda)
                low_lap, highs_lap = None, [None] * config.max_depth

            low_lap = torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[config.max_depth])).float().to(
                device) if low_lap is None else low_lap
            highs_lap = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[j])).float().to(device) if highs_lap[
                                                                                                                  j] is None else
                         highs_lap[j] for j in range(config.max_depth)]


            '''paths=glob.glob('feat_final/slat back*.npy')
            for path in paths:
              name=path.split('/')[-1].split('.')[0]
              low_lap=np.load(path)
              print (low_lap.shape, '')
              low_lap=torch.from_numpy(low_lap).cuda()
              
              voxels_pred = dwt_inverse_3d_lap((low_lap, highs_lap)).detach().cpu().numpy()
              vertices, traingles = mcubes.marching_cubes(voxels_pred[:,:,::8,::8,::8][0, 0], 0.0)
              vertices = (vertices.astype(np.float32) - 0.5) / (config.resolution/8) - 0.5
              mcubes.export_obj(vertices, traingles, name+'.obj')
            exit()'''


            text_input='four legs' 
            text = clip_tokenizer(text_input, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids']
            text=text.cuda()

            text_mani = clip_tokenizer(text_input, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids']
            text_mani=text_mani.cuda()

            text_empty=''
            text_empty = clip_tokenizer(text_empty, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids']
            text_empty=text_empty.cuda()
            
            
            #text_input2='short legs'


            if need_gt:
                voxels_pred = dwt_inverse_3d_lap((low_lap, highs_lap))
                vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
                vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
                mcubes.export_off(vertices, traingles, os.path.join(folder_path, f'{testing_sample_index}_gt.off'))

            if 1: #not evaluate_gt:

                highs_samples = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[i]), device=device) for i in
                                 range(config.max_depth)]

                data3np=np.load('swivel.npy')
                data3=torch.from_numpy(data3np).cuda()#.unsqueeze(0).unsqueeze(0)
                
                data3=data3
 


                voxels_input = dwt_inverse_3d_lap((data3, highs_samples))
                

                vertices, traingles = mcubes.marching_cubes(voxels_input[:,:,:,:,:].detach().cpu().numpy()[0, 0], 0.0)
                vertices = (vertices.astype(np.float32) - 0.5) /  (config.resolution) - 0.5
                mcubes.export_obj(vertices, traingles, 'input.obj')
                
                
                #import open3d as o3d
                #mesh = o3d.io.read_triangle_mesh('input.obj')
                import trimesh
                pcn = trimesh.load('input.obj.ply').vertices


                #maxi=torch.max(diffss)
                #scale=1.0/maxi
                #diffss_scale=diffss*scale



                



                mask=data3.clone()
                mask[:]=1

                
                


                selection=np.load('selection.npy')
                #print ('unique pcn', np.unique(pcn))
                
                max_size=max(max(pcn[:,0])-min(pcn[:,0]),max(pcn[:,1])-min(pcn[:,1]),max(pcn[:,2])-min(pcn[:,2]))
                
                
                scale=1.0/max_size
                
                pcn*=scale
                #print ('unique pcn scale', np.unique(pcn))
                
                pcn2=np.zeros((np.sum(selection),3))
                cntt=0
                for sel in range(2048):
                  if selection[sel]==1:
                    #print (sel, pcn[sel,:])
                    pcn2[cntt,:]=pcn[sel,:]
                    cntt+=1

                pcn3=[]
                for sel in range(pcn2.shape[0]):
                  pcn3.append((pcn2[sel,0],pcn2[sel,1],pcn2[sel,2]))
                some_array = np.array(pcn3, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32')])
                el = PlyElement.describe(some_array, 'vertex')
                PlyData([el]).write('seleted region.ply')



                #print ('unique pcn2', np.unique(pcn2))
                #exit()
                pcn_selection=(pcn2+0.5)*36+5
                #print ('unique pcn_selection', np.unique(pcn_selection))
                
                #print (cntt)
                #print ('0',np.unique(pcn2[:,0]))
                #print ('1',np.unique(pcn2[:,1]))
                #print ('2',np.unique(pcn2[:,2]))
                ##exit()

                #print (pcn2, pcn2.shape, 'pcn2')
                #print (torch.unique(pcn_selection))    
                pcn_selection=torch.from_numpy(pcn_selection).cuda().long()
                #print (torch.unique(pcn_selection), pcn_selection.shape)         
                xmin=torch.amin(pcn_selection[:,0])
                xmax=torch.amax(pcn_selection[:,0])
                ymin=torch.amin(pcn_selection[:,1])
                ymax=torch.amax(pcn_selection[:,1])
                zmin=torch.amin(pcn_selection[:,2])
                zmax=torch.amax(pcn_selection[:,2])
                
                
                print (xmin, xmax, ymin, ymax, zmin, zmax, mask.shape, '1111111111111111')


                '''xmin=13
                xmax=34
                ymin=8
                ymax=17
                zmin=13
                zmax=34'''
                

                mask[:,:,xmin:xmax,ymin:ymax,zmin:zmax]=0
                mask=erode(mask,3)
                #mask[:,:,xmin-3:xmax+3,ymin-3:ymax,zmin-3:zmax+3]=0            
                #mask[:]=1
                #mask[:,:,:,:23,:]=0

                #mask[torch.where(data3<=0)]=1



                #mask=diffss_scale.clone()
                #mask[torch.where(diffss_scale>=0.5)]=1 
                #mask[torch.where(diffss_scale<0.5)]=0

                '''mask=diffss.clone()
                mask[torch.where(diffss_scale>=0.5)]=1
                mask[torch.where(diffss_scale<0.5)]=0



                mask=dilate(mask,5)'''
                

                
                
                some_array=[]
                for i in range(0,46):
                    for j in range(0,46):
                        for k in range(0,46):
                            if mask[0,0,i,j,k]<0.5:
                                some_array.append((i/46-0.5,j/46-0.5,k/46-0.5,mask[0,0,i,j,k]*255,mask[0,0,i,j,k]*255,mask[0,0,i,j,k]*255))
                some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
                el = PlyElement.describe(some_array, 'vertex')
                PlyData([el]).write('init3_mask.ply')
                #exit()
                
                

                
                
                
                
                
                
                
                print ('generating...')


                text=text_input
                text = clip_tokenizer(text, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids']
                text=text.cuda()

                model_kwargs = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt'), 'text': text, 'start_step': 1,   'inpaint_image': data3, 'inpaint_mask': mask}
                

                text_mani=text_input
                
                text_mani = clip_tokenizer(text_mani, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids']
                text_mani=text_mani.cuda()
                
                
                

                def denoised_fn(x_start):
                    # Force the model to have the exact right x_start predictions
                    # for the part of the image which is known.
                    #return x_start
                    return (
                        x_start * (1 - model_kwargs['inpaint_mask'])
                        + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
                    )
                
                for repeat in range(0,3):



    
                    '''model_kwargs = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt'), 'text': text}
                    model_kwargs_empty = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt'), 'text': text_empty}


                    low_samples, model_outputs1 = diffusion_module.ddim_sample_loop(model=network,
                                                                    shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                                    device=device,
                                                                    clip_denoised=clip_noise, progress=True,
                                                                    noise=noise,
                                                                    eta=ddim_eta,
                                                                    model_kwargs=model_kwargs, model_kwargs_mani=model_kwargs, model_kwargs_empty=model_kwargs_empty)
                    low_samples=low_samples.detach()'''










                    setup_seed(repeat*10)
                    model_kwargs = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt'), 'text': text_mani, 'mask': mask,  'inpaint_image': data3, 'inpaint_mask': mask}
                    model_kwargs_mani = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt'), 'text': text_mani, 'mask': mask,  'start_step': 1,  'inpaint_image': data3, 'inpaint_mask': mask}
                    model_kwargs_empty = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt'), 'text': text_empty,  'inpaint_image': data3, 'inpaint_mask': mask}
                    







                    low_samples2, model_outputs2 = diffusion_module.ddim_sample_loop_mix(network2,
                                                                            shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                                            noise=noise,
                                                                            device=device,
                                                                            clip_denoised=clip_noise, 
                                                                            denoised_fn=denoised_fn,
                                                                            progress=True,
                                                                            eta=ddim_eta,
                                                                            data3=data3,
                                                                            model_kwargs=model_kwargs_mani, model_kwargs_mani=model_kwargs_mani,   model_kwargs_empty=model_kwargs_empty)





                    np.save('feat/'+text_input+str(repeat)+'.npy',low_samples2.detach().cpu().numpy())

                

                    if use_high_level_network:
                        upsampled_low = F.interpolate(low_samples, size=tuple(dwt_sparse_composer.shape_list[high_test_index]))
                        highs_samples[high_test_index] = network(upsampled_low)


                    '''voxels_pred = dwt_inverse_3d_lap((low_samples2, highs_samples))
                    vertices, traingles = mcubes.marching_cubes(voxels_pred[:,:,:,:,:].detach().cpu().numpy()[0, 0], 0.0)
                    vertices = (vertices.astype(np.float32) - 0.5) /  (config.resolution) - 0.5
                    mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'{m+start_index+extra_start}_'+str(repeat)+'_mani.obj'))'''
                    
                    
                    print (low_samples2.shape, data3.shape)

                    #low_samples2[:]=data3
                    #low_samples2[:,:,5:-5,23:35,18:-18]=low_samples[:,:,5:-5,23:35,18:-18]
                    #low_samples2[:,:,:,:23,:]=low_samples[:,:,:,:23,:]
                    #low_samples2[torch.where(low_samples2>0)]=low_samples[torch.where(low_samples2>0)]
                    voxels_pred = dwt_inverse_3d_lap((low_samples2, highs_samples))
                    vertices, traingles = mcubes.marching_cubes(voxels_pred[:,:,:,:,:].detach().cpu().numpy()[0, 0], 0.0)
                    vertices = (vertices.astype(np.float32) - 0.5) /  (config.resolution) - 0.5
                    mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'{m+start_index+extra_start}_'+str(repeat)+'_mani.obj'))

                    '''voxels_pred = dwt_inverse_3d_lap((low_samples, highs_samples))
                    vertices, traingles = mcubes.marching_cubes(voxels_pred[:,:,:,:,:].detach().cpu().numpy()[0, 0], 0.0)
                    vertices = (vertices.astype(np.float32) - 0.5) /  (config.resolution) - 0.5
                    mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'{m+start_index+extra_start}_'+str(repeat)+'_arms.obj'))'''



                if use_high_level_network and save_no_high:
                    highs_samples[high_test_index] = torch.zeros_like(highs_samples[high_test_index]).to(device)
                    voxels_pred = dwt_inverse_3d_lap((low_samples, highs_samples))
                    vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
                    vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
                    mcubes.export_obj(vertices, traingles, os.path.join(folder_path,
                                                                        f'{m + start_index + extra_start}_{testing_sample_index}_no_highs.obj'))

                print(f"Done {os.path.join(folder_path,f'{m+start_index+extra_start}_{testing_sample_index}.off')}!")

if __name__ == '__main__':

    debugger = MyDebugger(f'Network-Marching-Cubes-Diffusion-Gen',
                          is_save_print_to_file=False)

    from torch.multiprocessing import Pool
    torch.multiprocessing.set_start_method('spawn')

    GPU_CNT = 1
    PER_GPU_PROCESS = 1
    pool = Pool(GPU_CNT * PER_GPU_PROCESS)


    args = []
    assert testing_cnt % (GPU_CNT * PER_GPU_PROCESS) == 0
    if GPU_CNT * PER_GPU_PROCESS > 1:
        per_process_data_num = testing_cnt // (GPU_CNT * PER_GPU_PROCESS)
        for i in range(GPU_CNT):
            for j in range(PER_GPU_PROCESS):
                args.append((i, (i * PER_GPU_PROCESS + j) * per_process_data_num, per_process_data_num, debugger.file_path('.')))

        pool.map(one_generation_process, args)
    else:
        one_generation_process((0, 0, testing_cnt,  debugger.file_path('.')))

    print("done!")

