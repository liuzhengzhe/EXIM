from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import random
import os
import torch
import cv2
from im2mesh.common import (
    check_weights, get_tensor_values, transform_to_world,
    transform_to_camera_space, sample_patch_points, arange_pixels,
    make_3d_grid, compute_iou, get_occupancy_loss_points,
    get_freespace_loss_points
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from tqdm import tqdm
import logging
from im2mesh import losses

import importlib
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from im2mesh.dvr.models import depth_function
#    decoder, depth_function
#)
from im2mesh.common import (
    get_mask, image_points_to_world, origin_to_world, normalize_tensor)

from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from models.local_model import Discriminator
class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, train_loader, val_loader, train_dataset_shape, val_dataset_shape,  exp_name, cfg,optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        
        #print (optimizer)
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_dataset_shape = train_dataset_shape
        self.val_dataset_shape = val_dataset_shape



        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

        config_path = 'configs/config.py' #os.path.join(testing_folder, 'config.py')
        
        ## import config here
        spec = importlib.util.spec_from_file_location('*', config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        self.dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=3, wave=config.wavelet, mode=config.padding_mode).cuda()
        
        #self.mse=torch.nn.MSELoss()
        self.lambda_sparse_depth = 0
        
        self.use_cube_intersection=False
        self.occupancy_random_normal = False
        self.depth_range=[0, 2.4]
        self.lambda_depth=0.
        self.depth_from_visual_hull=False
        
        
        self.lambda_normal = 0.05

        #cfg = config.load_config('configs/', 'configs/default.yaml')
        depth_function_kwargs = cfg['model']['depth_function_kwargs']
        # Add the depth range to depth function kwargs
        depth_range = cfg['data']['depth_range']
        depth_function_kwargs['depth_range'] = self.depth_range
        self.call_depth_function = depth_function.DepthModule(
            **depth_function_kwargs)
            
            
   
        
        model_path_clip = "openai/clip-vit-large-patch14"
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
        
        
        self.criterion = torch.nn.BCELoss() 
        self.D = Discriminator(2048*3).to(device)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr = 1e-4)
        
        

    def train_step_shape(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss_shape(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
        
                
    def train_step(self,batch,it):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch,it)
        loss.backward()
        self.optimizer.step()

        return loss.item()




    def calc_photoconsistency_loss(self, mask_rgb, rgb_pred, img, pixels,
                                   reduction_method,  patch_size,
                                   eval_mode=False):
        ''' Calculates the photo-consistency loss.

        Args:
            mask_rgb (tensor): mask for photo-consistency loss
            rgb_pred (tensor): predicted rgb color values
            img (tensor): GT image
            pixels (tensor): sampled pixels in range [-1, 1]
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
            patch_size (int): size of sampled patch
            eval_mode (bool): whether to use eval mode
        '''
        if mask_rgb.sum() > 0: #self.lambda_rgb != 0 and mask_rgb.sum() > 0:
            batch_size, n_pts, _ = rgb_pred.shape
            loss_rgb_eval = torch.tensor(3)
            # Get GT RGB values
            #print ('img', img.shape, pixels.shape)
            rgb_gt = get_tensor_values(img, pixels)
            
            
            #print (rgb_pred.shape, rgb_gt.shape, mask_rgb.shape)
            
            '''for idx in range(rgb_pred.shape[0]):
              rgb_pred[idx][torch.where(mask_rgb[idx]==0)]=0
              rgb_gt[idx][torch.where(mask_rgb[idx]==0)]=0
              
              rgb_pred_im=rgb_pred[idx].detach().cpu().numpy()
              rgb_pred_im=np.reshape(rgb_pred_im,(64,64,3))
              rgb_gt_im=rgb_gt[idx].detach().cpu().numpy()
              rgb_gt_im=np.reshape(rgb_gt_im,(64,64,3))
              
              print (np.unique(rgb_pred_im),np.unique(rgb_gt_im))
              cv2.imwrite('pred'+str(idx)+'.png',rgb_pred_im*255)
              cv2.imwrite('gt'+str(idx)+'.png',rgb_gt_im*255)
            exit()'''
            

            
            
            
            # 3.1) Calculate RGB Loss
            #print  (rgb_pred.shape, rgb_gt.shape,mask_rgb.shape, 'rgbpred, gt, mask')
            #print ('loss',rgb_pred.shape, rgb_gt.shape, mask_rgb.shape, torch.unique(rgb_pred), torch.unique(rgb_gt), torch.unique(mask_rgb))
            loss_rgb = losses.l1_loss(
                rgb_pred[mask_rgb], rgb_gt[mask_rgb],
                reduction_method) * 1 / torch.sum(mask_rgb)


            
            #rgb_pred[torch.where(mask_rgb==False)]*=0.000001
            #rgb_pred[mask_rgb]=rgb_gt[mask_rgb]
            #rgb_gt[mask_rgb]*=0.000001
            
            
            #rgb_pred[mask_rgb]=rgb_pred[mask_rgb].detach()
            #rgb_gt[mask_rgb]=rgb_gt[mask_rgb].detach()





            return loss_rgb, rgb_pred, rgb_gt, mask_rgb
            #loss['loss'] += loss_rgb
            #loss['loss_rgb'] = loss_rgb
            if eval_mode:
                loss_rgb_eval = losses.l1_loss(
                    rgb_pred[mask_rgb], rgb_gt[mask_rgb], 'mean') * \
                    self.lambda_rgb

            # 3.2) Image Gradient loss
            if self.lambda_image_gradients != 0:
                assert(patch_size > 1)
                loss_grad = losses.image_gradient_loss(
                    rgb_pred, rgb_gt, mask_rgb, patch_size,
                    reduction_method) * \
                    self.lambda_image_gradients / batch_size
                loss['loss'] += loss_grad
                loss['loss_image_gradient'] = loss_grad
            if eval_mode:
                loss['loss_rgb_eval'] = loss_rgb_eval
                
    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
        img = data.get('img').to(device)
        #print ('img',img.shape, torch.unique(img))
        '''import cv2
        randint=random.randint(0,10000)
        
        cv2.imwrite(str(randint)+'.png', img[0,0,:,:].detach().cpu().numpy()*255)'''
        mask_img = data.get('img.mask').unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        #world_mat[0,:3,3]*=2
        #print (world_mat, 'word')
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        depth_img = data.get('img.depth', torch.empty(1, 0)
                             ).unsqueeze(1).to(device)
        #inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        # Get sparse point data
        if self.lambda_sparse_depth != 0:
            sparse_depth = {}
            sparse_depth['p'] = data.get('sparse_depth.p_img').to(device)
            sparse_depth['p_world'] = data.get(
                'sparse_depth.p_world').to(device)
            sparse_depth['depth_gt'] = data.get('sparse_depth.d').to(device)
            sparse_depth['camera_mat'] = data.get(
                'sparse_depth.camera_mat').to(device)
            sparse_depth['world_mat'] = data.get(
                'sparse_depth.world_mat').to(device)
            sparse_depth['scale_mat'] = data.get(
                'sparse_depth.scale_mat').to(device)
        else:
            sparse_depth = None

        return (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
                sparse_depth)


    def march_along_ray(self, ray0, ray_direction, c=None, it=None,
                        sampling_accuracy=None, text=None):
        ''' Marches along the ray and returns the d_i values in the formula
            r(d_i) = ray0 + ray_direction * d_i
        which returns the surfaces points.

        Here, ray0 and ray_direction are directly used without any
        transformation; Hence the evaluation is done in object-centric
        coordinates.

        Args:
            ray0 (tensor): ray start points (camera centers)
            ray_direction (tensor): direction of rays; these should be the
                vectors pointing towards the pixels
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        device =  torch.device("cuda" if 1 else "cpu")
        #print ('text training', text, it, sampling_accuracy)
        d_i = self.call_depth_function(text,ray0, ray_direction, self.model,
                                       c=c, it=it, n_steps=sampling_accuracy)

        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0

        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()

        # For sanity for the gradients
        d_hat = torch.ones_like(d_i).to(device)
        d_hat[mask_pred] = d_i[mask_pred]
        d_hat[mask_zero_occupied] = 0.

        return d_hat, mask_pred, mask_zero_occupied

    def pixels_to_world(self, pixels, camera_mat, world_mat, scale_mat, c,
                        it=None, text=None, sampling_accuracy=None, ):
        ''' Projects pixels to the world coordinate system.

        Args:
            pixels (tensor): sampled pixels in range [-1, 1]
            camera_mat (tensor): camera matrices
            world_mat (tensor): world matrices
            scale_mat (tensor): scale matrices
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        batch_size, n_points, _ = pixels.shape
        pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
                                             scale_mat)
        camera_world = origin_to_world(n_points, camera_mat, world_mat,
                                       scale_mat)
        ray_vector = (pixels_world - camera_world)

        d_hat, mask_pred, mask_zero_occupied = self.march_along_ray(
            camera_world, ray_vector, c, it, sampling_accuracy, text)
        p_world_hat = camera_world + ray_vector * d_hat.unsqueeze(-1)
        return p_world_hat, mask_pred, mask_zero_occupied




    def compute_loss_shape(self,batch):
        device = self.device

        p = batch.get('points').to(device)  #grid_coords
        occ = batch.get('occupancies').to(device)
        inputs = batch.get('inputs').to(device).unsqueeze(1)
        
        
        #print (inputs.shape, 'inputs')
        
        highs_samples=[torch.zeros((1, 1, 256, 256, 256)).cuda(), torch.zeros((1, 1, 136, 136, 136)).cuda(), torch.zeros((1, 1, 76, 76, 76)).cuda()]
        voxels_pred = self.dwt_inverse_3d_lap((inputs, highs_samples))
        
        
        voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)

        voxels_pred=torch.flip(torch.permute(voxels_pred, (0,1,4,3,2)),[2])
        
        inputs=voxels_pred[:,0,14:-14:2,14:-14:2,14:-14:2]
        
        
        #print (inputs.shape, 'inputs')


        # General points
        logits,_ = self.model(p,inputs,'', pred_occ=1, pred_color=0)
        

        loss_i=F.binary_cross_entropy_with_logits(logits, occ, reduction='none')# 
        
        #print (loss_i, 'loss_i')

        loss = loss_i.sum(-1).mean() 
        
        #print (loss, 'loss')
        return loss



    def compute_loss(self,batch,it):
        device = self.device




        #p = batch.get('points').to(device)
        #occ = batch.get('occupancies').to(device)
        inputs = batch.get('inputs').to(device).unsqueeze(1)
        
        
        #print (p.shape, occ.shape, inputs.shape, 'inputs')
        
        highs_samples=[torch.zeros((1, 1, 256, 256, 256)).cuda(), torch.zeros((1, 1, 136, 136, 136)).cuda(), torch.zeros((1, 1, 76, 76, 76)).cuda()]
        voxels_pred = self.dwt_inverse_3d_lap((inputs, highs_samples))
        
        
        voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)
        #voxels_pred/=0.9

        #voxels_pred=voxels_pred.detach().cpu().numpy()
        voxels_pred=torch.flip(torch.permute(voxels_pred, (0,1,4,3,2)),[2])
        
        inputs=voxels_pred[:,0,14:-14:2,14:-14:2,14:-14:2]

        # General points
        
        #print (p.shape, inputs.shape, 'shape')
        #print (torch.unique(p), p.shape, 'p shape')
        
        
        '''it=random.randint(0,10000)

        from plyfile import PlyData,PlyElement
        some_array=[]
        size=258
        print (p.shape)
        for i in range(p.shape[1]):
          if occ[0,i]==1:
            some_array.append((p[0,i,0],p[0,i,1],p[0,i,2]))
        some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32')])
        el = PlyElement.describe(some_array, 'vertex')
        
        PlyData([el]).write('p'+str(it)+'.ply')'''




        text = self.clip_tokenizer(batch.get('text'), padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids'].to(device)

        #print (p.shape, inputs.shape)
        '''logits,_ = self.model(p,inputs, text)
        loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean() '''



        with torch.no_grad():
          #loss = {}
          self.n_training_points=1024
          n_points = self.n_training_points #self.n_eval_points if eval_mode else self.n_training_points
          # Process data dictionary
          (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
            sparse_depth) = self.process_data_dict(batch)
            
          #print ('img',img.shape)
  
          # Shortcuts
          device = self.device
          patch_size = 1 #self.patch_size
          reduction_method = 'sum' #self.reduction_method
          batch_size, _, h, w = img.shape
          
          # Assertions
          assert(((h, w) == mask_img.shape[2:4]) and
                 (patch_size > 0) and
                 (n_points > 0))
  
          # Sample points on image plane ("pixels")
          #print (h,w, 'hw')
          if n_points >= h*w:
              p = arange_pixels((h, w), batch_size)[1].to(device)
          else:
              p = sample_patch_points(batch_size, n_points,
                                      patch_size=patch_size,
                                      image_resolution=(h, w),
                                      continuous=False, #self.sample_continuous,
                                      ).to(device)
          #p = arange_pixels((64,64), batch_size)[1].to(device)
          #print ('p', p.shape, p, torch.unique(p))
          #exit()
          #print ('rgb p.shape', p.shape)
          # Apply losses
          # 1.) Get Object Mask values and define masks for losses
          mask_gt = get_tensor_values(
              mask_img, p, squeeze_channel_dim=True).bool()
  
          # Calculate 3D points which need to be evaluated for the occupancy and
          # freespace loss
          p_freespace = get_freespace_loss_points(
              p, camera_mat, world_mat, scale_mat, self.use_cube_intersection,
              self.depth_range)
  
          depth_input = depth_img if (
              self.lambda_depth != 0 or self.depth_from_visual_hull) else None
          p_occupancy = get_occupancy_loss_points(
              p, camera_mat, world_mat, scale_mat, depth_input,
              self.use_cube_intersection, self.occupancy_random_normal,
              self.depth_range)
  
          # 2.) Initialize loss
          #loss['loss'] = 0
  
          # 3.) Make forward pass through the network and obtain predictions
          # with masks
          
  
  
          pixels=p
          c = inputs.detach() #self.encode_inputs(inputs)
          #print (c.shape, 'color',flush=True)
          # transform pixels p to world
          p_world, mask_pred, mask_zero_occupied = \
              self.pixels_to_world(pixels, camera_mat,
                                   world_mat, scale_mat, c, it, text=text)
          #_,rgb_pred = self.decode_color(p_world, c=c)
          
          
          #print ('rgb,p, world', p_world.shape)
  
        #print (p_world.shape, 'p world')
        '''it=random.randint(1,10000)
        from plyfile import PlyData,PlyElement
        some_array=[]
        size=258
        for i in range(p_world.shape[1]):
          #print (i)
          some_array.append((p_world[0,i,0],p_world[0,i,1],p_world[0,i,2]))
        some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32')])
        el = PlyElement.describe(some_array, 'vertex')
        
        PlyData([el]).write('p_world'+str(it)+'.ply')'''
        #exit()
        
        
          
        #print (torch.unique(p_world), p_world.shape, '')
        _,rgb_pred = self.model(p_world,inputs.detach(),text)

                
        '''a=np.zeros((256*256*3))
        rgb_pred_np=rgb_pred.detach().cpu().numpy()
        rgb_pred_np=rgb_pred.detach().cpu().numpy()
                
        print ('rgb mask', rgb_pred.shape, mask_pred.shape, mask_gt.shape)'''



        # 4.) Calculate Loss
        # 4.1) Photo Consistency Loss



        #rgb_gt = get_tensor_values(img, p)
        


        mask_rgb = mask_pred & mask_gt
        
        #print ('mask',mask_rgb.shape, mask_pred.shape, mask_gt.shape, torch.unique(mask_rgb), torch.unique(mask_pred))
        
        #print (mask_rgb.shape, mask_pred.shape, mask_gt.shape)
        loss_rgb,pred,gt,mask = self.calc_photoconsistency_loss(mask_rgb, rgb_pred,
                                        img, p, reduction_method, 
                                        patch_size, 0)
        


        if loss_rgb!=None:
          #print (loss_rgb.item())
          loss=loss_rgb*10














        return loss,pred,gt,mask

    def train_model(self):
        loss = 0
        start = self.load_checkpoint()

        for epoch in range(0, 500):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))


            train_data_loader_shape = self.train_dataset_shape.get_loader() #self.train_loader_shape




            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                
                
                
                
            for batch in train_data_loader_shape:
                loss = self.train_step_shape(batch)
                print("Current loss: {}".format(loss))
                sum_loss += loss
                
                
                
                
        for epoch in range(500, 700):              
            train_data_loader = self.train_loader #dataset.get_loader()
            it=0


            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                

            for batch in train_data_loader:
                it+=1
                #print ('batch',batch.shape)
                #loss = self.train_step(batch,it)
                
                #try:  
                self.model.zero_grad()
                self.model.train()
                self.optimizer.zero_grad()
                loss_reg, pred,gt,mask= self.compute_loss(batch,it)
                
                pred_input=torch.zeros(pred.shape).cuda()
                pred_input[torch.where(mask==True)]=pred[torch.where(mask==True)]

                gt_input=torch.zeros(gt.shape).cuda()
                gt_input[torch.where(mask==True)]=gt[torch.where(mask==True)]
                #pred[torch.where(mask==False)]*=0.000001

                D_output = self.D(pred_input)
                bs=pred.shape[0]
                y = torch.ones(bs, 1).cuda()
                G_loss = self.criterion(D_output, y)*0.1
                
                
                
                
                
                loss=loss_reg+G_loss

                loss.backward()

                self.optimizer.step()

                sum_loss += loss.item()
                #except:
                #  pass








                try:


                  self.D.zero_grad()
  
              
                  # train discriminator on real
                  y_real = torch.ones(bs, 1)
                  y_real = y_real.cuda()
              
                  D_output = self.D(gt_input.detach())
                  D_real_loss = self.criterion(D_output, y_real)
                  D_real_score = D_output
              
                  # train discriminator on facke
                  #z = Variable(torch.randn(bs, z_dim).to(device))
                  y_fake = torch.zeros(bs, 1).cuda()
              
                  D_output = self.D(pred_input.detach())
                  D_fake_loss = self.criterion(D_output, y_fake)
                  D_fake_score = D_output
              
                  # gradient backprop & optimize ONLY D's parameters
                  D_loss = D_real_loss*0.1 + D_fake_loss*0.1
                  D_loss.backward()
                  self.D_optimizer.step()
                      
  
                  print("Current loss: {},{},{}".format(loss_reg.item(), G_loss.item(), D_loss.item()))
                except:
                  pass
                
            try:  
              self.writer.add_scalar('training loss last batch', loss.item(), epoch)
              self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
            except:
              pass


    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        name=path.split('/')[-1].split('_')[-1].split('.')[0]
        root='/'.join(path.split('/')[:-1])
        if int(name)%10!=0:
          try:
              os.remove(root+'/checkpoint_epoch_'+str(int(name)-30)+'.tar')
          except:
              pass
        if not os.path.exists(path):
            torch.save({'epoch':epoch,'model_state_dict': self.model.state_dict(),'D': self.D.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('!!!!!!!!!!!!!!Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #,strict=False)
        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_loader.__iter__() #self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss( val_batch).item()

        return sum_val_loss / num_batches
