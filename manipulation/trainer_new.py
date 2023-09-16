import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from utils.debugger import MyDebugger
from utils.other_utils import save_off, save_pointclouds_fig
from utils.meter import Meter
from models.network import MultiScaleMLP, SparseComposer, create_coordinates, Discriminator
#from models.module.diffusion_network import MyUNetModel, UNetModel
from diffusers_mani import AutoencoderKL, UNet3DConditionModel
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from models.module.resample import UniformSampler, LossSecondMomentResampler, LossAwareSampler
from data.data import SDFSamples
from utils.other_utils import TVLoss
from tqdm import tqdm
import os, clip
import argparse
import mcubes
from modelAE import IM_AE
import random
from plyfile import PlyData,PlyElement


import torch
path='../data/model_epoch_3_1450.pth'
model=torch.load(path)
model2=torch.load(path)
#for key in model.keys():
#  print (key, model[key].shape)
zero=torch.zeros((32,2,3,3,3)).cuda()
model['conv_in.weight']=torch.cat((model['conv_in.weight'], zero),1)
#model2['module.'+key]=  model[key]
#del model2[key]
torch.save(model, 'init.pth')
  

def dilate(img,ksize=5):
  p=(ksize-1)//2
  img=F.pad(img, pad=[p,p,p,p,p,p], mode='reflect')
  out=F.max_pool3d(img, kernel_size=ksize, stride=1, padding=0)
  return out

def erode(img, ksize=5):
  out=1-dilate(1-img, ksize)
  return out#, 1-img, dilate(1-img, ksize)

def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Trainer(object):

    def __init__(self, config, debugger):
        self.debugger = debugger
        self.config = config
        self.out_dim = self.config.out_dim if hasattr(self.config, 'out_dim') else 3

        self.im_ae = IM_AE()         
        model_dir = '../data/IM-NET-pytorch.pth' #fin.readline().strip()
        self.im_ae.im_network.load_state_dict(torch.load(model_dir), strict=True)
        self.im_ae.im_network.eval()


    def adjust_learning_rate(self, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.config.scheduler[i]['Initial'] * (self.config.scheduler[i]['Factor'] ** (epoch // self.config.scheduler[i]['Interval'] ))

    def train_network(self):

        ### create dataset

        samples = SDFSamples(data_path = self.config.data_path,
                             data_folder = self.config.data_folder,
                             resolution = self.config.resolution,
                             num_points = self.config.num_points,
                             interval = self.config.interval,
                             first_k = self.config.first_k,
                             use_surface_samples = self.config.use_surface_samples,
                             sample_resolution = self.config.sample_resolution,
                             sample_ratio = self.config.sample_ratio,
                             load_ram = self.config.load_ram,
                             use_preload=self.config.use_preload if hasattr(self.config, 'use_preload') else False,
                             data_files=self.config.data_files if hasattr(self.config, 'data_files') else [],
                             )

        ### level_indices_remap
        level_map = {idx : level for idx, (_, level) in enumerate(self.config.data_files)}

        data_loader = DataLoader(dataset = samples,
                                 batch_size = self.config.batch_size,
                                 num_workers = self.config.data_worker,
                                 shuffle = True,
                                 drop_last = True)
        


        ### initialize network
        dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
        dwt_forward_3d_lap = DWTForward3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device) #sdf->freq 46,76
        composer_parms = dwt_inverse_3d_lap if config.use_dense_conv else None
        dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution], J=config.max_depth,
                                             wave=config.wavelet, mode=config.padding_mode, inverse_dwt_module=composer_parms).to(
            device)

        if self.config.training_stage == self.config.max_depth:

            network = UNet3DConditionModel() #.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)


            '''network = UNetModel(in_channels=1,
                      model_channels=self.config.unet_model_channels,
                      out_channels=2 if hasattr(self.config,
                                                'diffusion_learn_sigma') and self.config.diffusion_learn_sigma else 1,
                      num_res_blocks=self.config.unet_num_res_blocks,
                      channel_mult=self.config.unet_channel_mult_low,
                      attention_resolutions=self.config.attention_resolutions,
                      dropout=0,
                      dims=3,
                      activation=self.config.unet_activation if hasattr(self.config, 'unet_activation') else None)'''
        else:
            network = UNet3DConditionModel() 
            '''network = MyUNetModel(in_channels= 1,
                                spatial_size= dwt_sparse_composer.shape_list[self.config.training_stage][0],
                                model_channels=self.config.unet_model_channels,
                                out_channels= 1,
                                num_res_blocks=self.config.unet_num_res_blocks,
                                channel_mult=self.config.unet_channel_mult,
                                attention_resolutions=self.config.attention_resolutions,
                                dropout=0,
                                dims=3)'''
        #print (network)
        network.to('cuda')
        ### diffusion setting
        betas = get_named_beta_schedule(self.config.diffusion_beta_schedule, self.config.diffusion_step, self.config.diffusion_scale_ratio)
        diffusion_module = GaussianDiffusion(betas=betas,
                                      model_var_type=self.config.diffusion_model_var_type,
                                      model_mean_type=self.config.diffusion_model_mean_type,
                                      loss_type=self.config.diffusion_loss_type,
                                      rescale_timesteps=self.config.diffusion_rescale_timestep if hasattr(self.config, 'diffusion_rescale_timestep') else False)
        self.config.vis_results = False ## cannot visualize as too slow

        ## sample
        if self.config.diffusion_sampler == 'uniform':
            sampler = UniformSampler(diffusion_module)
        elif self.config.diffusion_sampler == 'second-order':
            sampler = LossSecondMomentResampler(diffusion_module)
        else:
            raise Exception("Unknown Sampler.....")

        ## only convert all
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            network = nn.DataParallel(network)

        network = network.to(device)



        ## reload the network if needed

        if self.config.network_resume_path is not None:
            ### remove something that is not needed
            network_state_dict = torch.load(self.config.network_resume_path)
            new_state_dict = network.state_dict()
            for key in list(network_state_dict.keys()):
                
                if key not in new_state_dict:
                    del network_state_dict[key]
            
            network.load_state_dict(network_state_dict,strict=True)
            network.train()
            print(f"Reloaded thenetwork from {self.config.network_resume_path}")


        log_meter = Meter()
        log_meter.add_attributes('mse_loss')
        log_meter.add_attributes('total_loss')
        mse_fuction = self.config.loss_function



        ##### MAIN #####
        stage = self.config.training_stage

        ### reloaded optimizer for each stage
        #### initialize
        if hasattr(self.config, 'optimizer') and self.config.optimizer:
            optimizer = self.config.optimizer(network.parameters(), lr = self.config.lr,
                                              betas=(self.config.beta1, self.config.beta2))
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, self.config.beta2)
                                         )

        if self.config.lr_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=self.config.lr_decay_rate)
        else:
            scheduler = None

        if self.config.optimizer_resume_path is not None:
            optimizer_state_dict = torch.load(self.config.optimizer_resume_path)
            new_state_dict = optimizer.state_dict()
            for key in list(optimizer_state_dict.keys()):
                if key not in new_state_dict:
                    del optimizer_state_dict[key]
            optimizer.load_state_dict(optimizer_state_dict)
            print(f"Reloaded the optimizer from {self.config.optimizer_resume_path}")
            self.config.optimizer_resume_path = None

        # mixed precision training
        if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
            scaler = GradScaler()

            if hasattr(self.config, 'scaler_resume_path') and self.config.scaler_resume_path is not None:
                scaler_state_dict = torch.load(self.config.scaler_resume_path)
                scaler.load_state_dict(scaler_state_dict)



        for idx in range(self.config.starting_epoch, self.config.training_epochs + 1):

            # add this
            if scheduler is not None:
                scheduler.step(idx)

            with tqdm(data_loader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {idx}')

                network.train()
                ## main training loop
                for data,  text, code_indices, part in tepoch:
                    ## remove gradient
                    optimizer.zero_grad()

                    low_lap, highs_lap = None, [None] * self.config.max_depth
                    coeff_gt = data
                    for j, gt in enumerate(coeff_gt):
                        #print (j)
                        level = level_map[j]
                        if level == self.config.max_depth:
                            low_lap = coeff_gt[j].unsqueeze(1).to(device)
                        else:
                            highs_lap[level] = coeff_gt[j].unsqueeze(1).to(device)
                        #print (low_lap, highs_lap)
                    code_indices = code_indices.to(device)

                    loss = 0
                    ###
                    mse_loss = 0.0


                    highs_lap = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[j])).float().to(device) if highs_lap[
                                                                                                                  j] is None else
                         highs_lap[j] for j in range(config.max_depth)]
                         
                    #print (low_lap.shape, highs_lap, flush=True)
                    #exit()
                    '''voxels_input = dwt_inverse_3d_lap((low_lap, highs_lap))
                    batch_voxels = voxels_input.clone()
                    batch_voxels[torch.where(voxels_input<=0)]=1
                    batch_voxels[torch.where(voxels_input>0)]=0   
                    
                    
                    model_z,_ = self.im_ae.im_network(batch_voxels[:,:,2:-1:4,2:-1:4,2:-1:4], None, None, is_training=False)
                    model_float = self.im_ae.z2voxel(model_z)[1:-1,1:-1,1:-1].unsqueeze().unsqueeze()
                    model_float=torch.nn.functional.interpolate(model_float.float().unsqueeze(0).unsqueeze(0), size=46, mode='nearest').long()[0,0,:,:,:]
                    model_float_small=torch.nn.functional.interpolate(model_float.float().unsqueeze(0).unsqueeze(0), size=36, mode='nearest').long()#[0,0,:,:,:]
                    model_float[:]=-1
                    model_float[:,:,5:-5,5:-5,5:-5]=model_float_small'''
                    
                    
                    
                    
                    
                    
                    
                    inpaint_mask=torch.ones(low_lap.shape).cuda()   
                    
                    if random.uniform(0, 1)>0.0:

                    
                      part=part.unsqueeze(1)                                       
                      
                      cate=torch.max(part)
  
                      rand_num=random.randint(1, cate-1)
                      
                      for r in range(rand_num):
  
    
                        rand_cate=random.randint(0, cate)
    
                   
                        #print ('rand_cat',rand_cate,inpaint_mask.shape )
                        inpaint_mask[torch.where(part==rand_cate)]=0
                      
                      #print ('inpaint_mask2',inpaint_mask.shape, torch.unique(inpaint_mask))
                      
                      
                      
                      
  
                      
                      
                      #inpaint_mask = F.dropout(inpaint_mask, p=0.002)   
                      #inpaint_mask[torch.where(inpaint_mask>0)] =1                  
                      
                      rand_dilate=random.randint(0,2)*2+1
  
                      #print ('inpaint_mask2.5',inpaint_mask.shape, torch.unique(inpaint_mask))
                      inpaint_mask=erode(inpaint_mask,rand_dilate)
                      
                      
                      
  
                      
                      #inpaint_mask=torch.ones(inpaint_mask.shape).cuda()-inpaint_mask
                      
                      
                      #print ('inpaint_mask3',inpaint_mask.shape, torch.unique(inpaint_mask))
                      
                      
                      
                      #print ('part',torch.unique(part), part.shape)     
  
  
                      '''some_array=[]
                      for i in range(0,46):
                          for j in range(0,46):
                              for k in range(0,46):
                                  if part[0,0,i,j,k]==0:
                                    some_array.append((i/46-0.5,j/46-0.5,k/46-0.5,part[0,0,i,j,k]*50,part[0,0,i,j,k]*50,part[0,0,i,j,k]*50))
                      some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
                      el = PlyElement.describe(some_array, 'vertex')
                      PlyData([el]).write('part.ply')
  
                      
                      some_array=[]
                      for i in range(0,46):
                          for j in range(0,46):
                              for k in range(0,46):
                                  if part[0,0,i,j,k]>-1:
                                    some_array.append((i/46-0.5,j/46-0.5,k/46-0.5,inpaint_mask[0,0,i,j,k]*255,inpaint_mask[0,0,i,j,k]*255,inpaint_mask[0,0,i,j,k]*255))
                      some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
                      el = PlyElement.describe(some_array, 'vertex')
                      PlyData([el]).write('inpatint_mask.ply')
                      exit()'''
                      
                    
                    
                    #else:
                    #    inpaint_mask[:]=0
                    
                    for b in range(low_lap.shape[0]):
                      
                      whe=torch.where(inpaint_mask[b:b+1]==0)
                      if whe[0].shape[0]!=0:
                        #print (whe)
                        xmin=torch.amin(whe[2])
                        xmax=torch.amax(whe[2])
                        ymin=torch.amin(whe[3])
                        ymax=torch.amax(whe[3])
                        zmin=torch.amin(whe[4])
                        zmax=torch.amax(whe[4])
                        #print (xmin,xmax,ymin,ymax,zmin,zmax)
                        inpaint_mask[b:b+1][:,:,xmin:xmax,ymin:ymax,zmin:zmax]=0
                      
                    
                    


                    if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                        with autocast():
                            
                            mse_loss = self.gt_supervision(diffusion_module, highs_lap, low_lap, text, inpaint_mask, 
                                                           mse_fuction, mse_loss, network, sampler, stage, dwt_sparse_composer.shape_list[stage])
                    else:
                        mse_loss = self.gt_supervision(diffusion_module, highs_lap, low_lap,text,inpaint_mask,
                                                       mse_fuction, mse_loss, network, sampler, stage, dwt_sparse_composer.shape_list[stage])


                    log_meter.add_data('mse_loss', mse_loss.item())
                    loss = loss + mse_loss
                    log_meter.add_data('total_loss', loss.item())

                    if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self.config.use_gradient_clip:
                        torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip_value)

                    if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()


                    log_dict = log_meter.return_avg_dict()

                    tepoch.set_postfix(**log_dict)

                self.print_dict(idx, log_dict)
                log_meter.clear_data()



            ## saving the models
            if idx % self.config.saving_intervals == 0:
                # save
                network_resume_path = self.debugger.file_path(f'model_epoch_{stage}_{idx}.pth')
                optimizer_resume_path = self.debugger.file_path(f'optimizer_epoch_{stage}_{idx}.pth')
                torch.save(network.state_dict(), network_resume_path)
                torch.save(optimizer.state_dict(), optimizer_resume_path)
                #try:
                #    idx2=int(idx-self.config.saving_intervals*3)
                #    os.remove(self.debugger.file_path('optimizer_epoch_3_'+str(idx2)+'.pth'))
                #    os.remove(self.debugger.file_path('model_epoch_3_'+str(idx2)+'.pth'))
                #except:
                #    pass
                MyDebugger.save_text(idx, 'network_resume_path', network_resume_path)
                MyDebugger.save_text(idx, 'optimizer_resume_path', optimizer_resume_path)
                MyDebugger.save_text(idx, 'starting_stage', stage)

                if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                    scaler_resume_path = self.debugger.file_path(f'scaler_epoch_{stage}_{idx}.pth')
                    torch.save(scaler.state_dict(), scaler_resume_path)
                    MyDebugger.save_text(idx, 'scaler_resume_path', scaler_resume_path)


    def gt_supervision(self, diffusion_module, highs_lap, low_lap, feature, inpaint_mask,mse_fuction, mse_loss, network,
                       sampler, stage, spatial_shape):

        for j in range(self.config.max_depth + 1):

            ## no need to train for zero and gt
            if j != stage:
                continue

            if j == self.config.max_depth:
                #import pdb;pdb.set_trace()
                t, weights = sampler.sample(low_lap.size(0), device=device) #lzz1: low_lap ->256*256*256
                iterative_loss = diffusion_module.training_losses(model=network, x_start=low_lap, t=t, text=feature, inpaint_mask=inpaint_mask)
                mse_loss = mse_loss + torch.mean(iterative_loss['loss'] * weights)

            else:
                coeff_gt = highs_lap[stage]
                upsampled_low = F.interpolate(low_lap, size=tuple(spatial_shape))
                coeff = network(upsampled_low)
                mse_loss = mse_loss + mse_fuction(coeff, coeff_gt)



        return mse_loss

    def print_dict(self, idx, record_dict : dict):

        str = f'Epoch {idx} : '
        for key, item in record_dict.items():
            str += f'{key} : {item} '

        print(str)



if __name__ == '__main__':
    import importlib
    #torch.multiprocessing.set_start_method('spawn')  # good solution !!!!

    ## additional args for parsing
    optional_args = [("network_resume_path", str), ("optimizer_resume_path", str) ,("starting_epoch", int), ('starting_stage', int),
                     ("special_symbol", str), ("resume_path", str), ("discriminator_resume_path", str), ("discriminator_opt_resume_path", str),
                     ("scaler_resume_path", str)]

    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)

    args = parser.parse_args()
    ## Resume setting
    resume_path = None

    ## resume from path if needed
    if args.resume_path is not None:
        resume_path = args.resume_path

    if resume_path is None:
        from configs import config
        resume_path = os.path.join('configs', 'config.py')
    else:
        ## import config here
        spec = importlib.util.spec_from_file_location('*', resume_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)


    for optional_arg, arg_type in optional_args:
        if args.__dict__.get(optional_arg, None) is not None:
            locals()['config'].__setattr__(optional_arg, args.__dict__.get(optional_arg, None))


    debugger = MyDebugger(f'Wavelet-Training-experiment{"-" + config.special_symbol if len(config.special_symbol) > 0 else config.special_symbol}', is_save_print_to_file = True, config_path = resume_path)
    trainer = Trainer(config = config, debugger = debugger)
    trainer.train_network()

