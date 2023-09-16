import data_processing.implicit_waterproofing as iw
import mcubes
import trimesh
import torch
import os
from glob import glob
import numpy as np
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
class Generator(object):
    def __init__(self, model, threshold, exp_name, checkpoint = None, device = torch.device("cuda"), resolution = 16, batch_points = 1000000):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.resolution = resolution
        self.checkpoint_path = 'experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/checkpoints/' #os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format( exp_name)
        self.load_checkpoint(checkpoint)
        self.batch_points = batch_points

        self.min = -0.5
        self.max = 0.5


        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = self.max + self.min
        b = self.max - self.min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)
        model_path_clip = "openai/clip-vit-large-patch14"
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)

    def estimate_colors(self, vertices, c=None,text=None,noise=None):
        ''' Estimates vertex colors by evaluating the texture field.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): latent conditioned code c
        '''


        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, 100000)
        colors = []
        for vi in vertices_split:
            vi = vi.to(device)
            #print ('vi, c', torch.unique(vi),vi.unsqueeze(0).shape,c.shape)
            with torch.no_grad():
                _, ci = self.model(vi.unsqueeze(0),c,text,pred_occ=0,pred_color=1,noise=noise)

                ci=ci.detach().cpu().numpy()[0]
                #print (ci.shape, 'ci')
                #ci=ci[:,::-1]
                #ci = self.model.decode_color(
                #    vi.unsqueeze(0), c).squeeze(0).cpu()
                #ci[:]=1
            colors.append(ci)
            #print ('colors', ci.shape, np.unique(ci))
        colors = np.concatenate(colors, axis=0)
        #print (colors.shape, 'colors')
        colors = np.clip(colors, 0, 1)
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([
            colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)],
            axis=1)
        return colors
    def generate_mesh(self, data,text,noise=None):


        inputs = data['inputs'].to(self.device)
        inputs2 = data['inputs2'].to(self.device)

        logits_list = []
        color_list = [] 
        #text='this is a blue chair'
        text_str=text
        text = self.clip_tokenizer(text_str, max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids'].cuda()
        
        #self.model.train()
        


        for points in self.grid_points_split:
            #print (torch.unique(points))    
            with torch.no_grad():
                logits,_ = self.model(points,inputs,text,pred_occ=1,pred_color=0)
                #_,color = self.model(points,inputs2,text,pred_occ=0,pred_color=1,noise=noise)
            logits_list.append(logits.squeeze(0).detach().cpu())
            #color_list.append(color.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0)
        #color = torch.cat(color_list, dim=0)

        logits = np.reshape(logits.numpy(), (self.resolution,)*3)
        #color = np.reshape(color.numpy(), (self.resolution,self.resolution,self.resolution,3))


        '''from plyfile import PlyData,PlyElement
        some_array=[]
        for i in range(0,256,8):
          for j in range(0,256,8):
            for k in range(0,256,8):
                #if logits[i,j,k]>0.0:
                #print (color[i,j,k,0]*255,color[i,j,k,1]*255,color[i,j,k,2]*255)
                some_array.append((i,j,k,color[i,j,k,0]*255,color[i,j,k,1]*255,color[i,j,k,2]*255))
        some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'), ('red', 'uint8'), ('green', 'uint8'),    ('blue', 'uint8')])
        el = PlyElement.describe(some_array, 'vertex')
        
        
        PlyData([el]).write('voxel'+text_str[:20]+'.ply')
        print ('voxel')
        #print ('np', np.unique(color),color.shape)
        #exit()'''
         

        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        #print (np.log(self.threshold), np.log(1. - self.threshold), threshold, 'threshold')
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        #print (vertices.shape, np.unique(np.array(vertices)), 'vertices')
        '''vertices[:,0]*=(-1)
        vertices[:, [2,1,0]] = vertices[:, [0,1,2]]
        vertices/=256 #'''
        
        
        vertices[:, [2,1,0]] = vertices[:, [0,1,2]]
        
        vertices=(vertices-0.5)/128-1#
        
        
        #print ( logits.shape, self.resolution, 'resolu')

        #print (vertices.shape, np.unique(np.array(vertices)), 'vertices')

        vertex_colors = self.estimate_colors(np.array(vertices), inputs2,text,noise=noise)

        
        
        '''text_str='has pink seat'
        text = self.clip_tokenizer(text_str, max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)['input_ids'].cuda()
        
        

        vertex_colors2 = self.estimate_colors(np.array(vertices), inputs2,text)
        
        
        for i in range(vertices.shape[0]):
          v=vertices[i]
          if v[1]<0.04 and v[1]>0.03:  #0 front back #1: up&down <0:down. >0:up
            vertex_colors[i,:]=vertex_colors2[i,:]'''
        
        
        '''import open3d as o3d



        mesh = o3d.io.read_triangle_mesh('mani/it has thick backrest1.npy_it is a orange chair.obj')
        vertex_colors1=np.asarray(mesh.vertex_colors)
        vertices1=np.asarray(mesh.vertices)

        mesh = o3d.io.read_triangle_mesh('mani/it has thick backrest1.npy_it is yellow in color.obj')
        vertex_colors2=np.asarray(mesh.vertex_colors)
        vertices2=np.asarray(mesh.vertices)
        
        print (vertex_colors1, vertex_colors2)
        print (vertex_colors1.shape, vertex_colors2.shape)
        
        
        for i in range(vertices1.shape[0]):
          v=vertices1[i]
          if v[1]<-0.1:  #0 front back #1: up&down <0:down. >0:up
            vertex_colors1[i,:]=vertex_colors2[i,:]
          #else:
          #  print (vertex_colors.shape, vertex_colors1.shape)
          #  vertex_colors[i,:]=vertex_colors1[i,:]
        
        vertices=vertices1
        vertex_colors=vertex_colors1
        
        #exit()
        #vertex_colors = '''
        




        #vertices -= 1

        #rescale to original scale
        
        #vertices=(vertices-0.5)/256-0.5
        
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=triangles,
            #vertex_normals=mesh.vertex_normals,
            vertex_colors=vertex_colors, 
            process=False)
        

        
        #trimesh.exchange.export.export_mesh(mesh, 'color.obj')
        #print ('1',np.unique(vertices))
        
        #vertices=(vertices-0.5)/256-0.5
        #remove translation due to padding

        
        #print ('2',np.unique(vertices))
        
        '''step = (self.max - self.min) / (self.resolution - 1)
        
        print ('1',np.unique(vertices))
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]'''
        
        
        
        





        #voxels_pred=torch.nn.functional.interpolate(voxels_pred,scale_factor=1/0.9)
        #vertices*=0.9
        

        

        #mcubes.export_obj(vertices, traingles, 'mesh/'++'low.obj')
        #mesh = trimesh.Trimesh(vertices, triangles)
        return mesh,vertices, triangles

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)
        
        #print (np.unique(vertices))
        

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]
        
        
        #print (np.unique(vertices))                        

        return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path+'/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        print (checkpoint['model_state_dict'].keys())
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)  