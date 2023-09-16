# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import os


clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44

import trimesh,cv2

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.utils import save_frame
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

from scene_synthesis.utils import get_textured_objects


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


def floor_plan_renderable(room, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces = room.floor_plan
    # Center the floor
    vertices -= room.floor_plan_centroid
    # Return a simple-3dviz renderable
    return Mesh.from_faces(vertices, faces, color)


def floor_plan_from_scene(
    scene,
    path_to_floor_plan_textures,
    without_room_mask=False
):
    if not without_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
    else:
        room_mask = None
    # Also get a renderable for the floor plan
    floor, tr_floor,im, offset_scale = get_floor_plan(
        scene,
        [
            os.path.join(path_to_floor_plan_textures, fi)
            for fi in os.listdir(path_to_floor_plan_textures)
        ]
    )
    return [floor], [tr_floor], room_mask,im, offset_scale


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )
    
    #print ('vertices',floor.vertices)


    #vertices+=2.5
    #vertices*=50

    #vertices=np.concatenate((vertices.vertices[:,0:1],raw_mesh.vertices[:,2:3]),1)
    #vertices=raw_mesh.vertices
    
    



    #vertices+=2.5
    #vertices*=50

    xmin=np.amin(vertices[:,0])
    xmax=np.amax(vertices[:,0])
    ymin=np.amin(vertices[:,2])
    ymax=np.amax(vertices[:,2])
    
    
    offset1=xmin
    offset2=ymin
    
    
        
    vertices[:,0]-=offset1
    vertices[:,2]-=offset2
    
    
    #print ('vertices',vertices)
    
    long_length=max(xmax-xmin, ymax-ymin)
    scale=256/long_length
    
    vertices[:,0]*=scale
    vertices[:,2]*=scale
    
    
    '''from plyfile import PlyData,PlyElement
    some_array=[]
    for i in range(floor.vertices.shape[0]):
      some_array.append((vertices[i,0],vertices[i,1],vertices[i,2]))
    some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32')])
    el = PlyElement.describe(some_array, 'vertex')
    PlyData([el]).write('floor.ply')'''
    
    vertices=vertices.astype('int')
    
    #print ('vertices',vertices)
    
    im=np.zeros((256,256,3))
    
    
    #print (faces, faces.shape, 'face')
    
    
    
    for face in faces:
      v=[]
      i=face[0]
      j=face[1]
      k=face[2]
      v.append([vertices[i][2],vertices[i][0]])
      v.append([vertices[j][2],vertices[j][0]])
      v.append([vertices[k][2],vertices[k][0]])
      v=np.asarray(v)
      cv2.fillPoly(im,pts=[v], color=(255,255,255))
      
    #cv2.imwrite('a.png',im)
    #exit()
    

    return floor, tr_floor, im, [offset1, offset2, scale]


'''from scipy import signal
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d'''


'''def mapping(cate):
    if cate<=12 or cate==35: #cabinate
      return 0
    if cate>=13 and cate<=17 or cate==36: #bed
      return 1
    if cate<=22 or cate==32 or cate==37 or cate==38: #chair
      return 2
    if cate<=25 or cate==39: #table
      return 3
    if cate<=31 or cate==40:  #sofa
      return 4
    if cate<=33 or cate==41 or cate==42: #light
      return 5
'''
















def get_textured_objects_in_scene(scene, category, offset_scale, class_labels, category_dic, category_list, ignore_lamps=False):
    renderables = []
    
    semantic=np.zeros((256,256,3))
    height=np.zeros((256,256,3))
    height2=np.zeros((256,256,3))
    orientation=np.zeros((256,256,3))
    instance=np.zeros((256,256))
    instance_lamp=np.zeros((256,256))  
    
    feature_map= np.zeros((32,32,768))  
    feature_map_lamp = np.zeros((32,32,768))
    
    object_count=[0,0,0,0,0,0]


    paths=[]
    for idx in range(len(scene.bboxes)):
        furniture=scene.bboxes[idx]
        model_path = furniture.raw_model_path
        
        
        
        paths.append(model_path)
        
        continue
        
        
        image_path='/'.join(model_path.split('/')[:-1])+'/image.jpg'
        
        
        image = Image.open(image_path)

        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        image_features = clip_model.get_image_features(**inputs)
        
        image_features=image_features/image_features.norm(dim=-1, keepdim=True)
        
        image_features=image_features.unsqueeze(0)
        
        #print ('image_features', image_features.shape, torch.norm(image_features,p=2,dim=-1))
        
        
        
        
        
        
        if not model_path.endswith("obj"):
            import pdb
            pdb.set_trace()

        # Load the furniture and scale it as it is given in the dataset
        raw_mesh = TexturedMesh.from_file(model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)
        
        #print (raw_mesh, 'rawmesh', raw_mesh.vertices)






        
        
        vertices=raw_mesh.vertices
        
        



        #vertices+=2.5
        #vertices*=50
        
        
        #print ('off set scale', offset_scale)
        
        vertices[:,0]-=offset_scale[0]
        vertices[:,2]-=offset_scale[1]
        vertices[:,:]*=offset_scale[2]
        


        '''from plyfile import PlyData,PlyElement
        some_array=[]
        for i in range(vertices.shape[0]):
          some_array.append((vertices[i,0],vertices[i,1],vertices[i,2]))
        some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32')])
        el = PlyElement.describe(some_array, 'vertex')
        PlyData([el]).write('object'+str(idx)+'.ply')'''



        #print (category[idx], np.where(category[idx]==1), np.where(category[idx]==1)[0], 14-np.where(category[idx]==1)[0][0])
        
        
        #print ('category', category[idx], idx)
        cate=np.where(category[idx]==1)[0][0]
        
        
        
        #print ('category',category_list)
        #print ('cate',cate)
        
        category_name=category_list[cate]
        #print (category_name)


        if category_name in category_dic.keys():
          cate=category_dic[category_name]        
        else:
          if 'cabinet' in category_name.lower():
            cate=0
          elif 'bed 'in category_name.lower():
            cate=1
          elif 'chair 'in category_name.lower():
            cate=2
          elif 'table 'in category_name.lower():
            cate=3
          elif 'sofa 'in category_name.lower():
            cate=4
          elif 'light 'in category_name.lower():
            cate=5
          else:
            continue
        
        #print (cate)
        
        #print (cate,'category')
        
        object_count[cate]+=1

    
        xmin=max(0,int(np.amin(vertices[:,0]))+2)
        xmax=max(xmin+1,min(255,int(np.amax(vertices[:,0]))-2))
        ymin=max(0,int(np.amin(vertices[:,2]))+2)
        ymax=max(ymin+1,min(255,int(np.amax(vertices[:,2]))-2))


        
        #print ('semantic',category, cate, cate,(cate+1)*40)
        
        
        
        
        
        #print ('theta',np.unique(theta)) [-3.14, 3.14]
        if cate!=5:
          semantic[xmin:xmax,ymin:ymax,:]=(cate+1)*40
          height[xmin:xmax,ymin:ymax,:]=np.amax(vertices[:,1])
          height2[xmin:xmax,ymin:ymax,:]=np.amin(vertices[:,1])
          orientation[xmin:xmax,ymin:ymax,:]=(theta+3.14)*250/3.14/2
          
          feature_map[int(xmin/8):int(xmax/8),int(ymin/8):int(ymax/8),:]=image_features.detach().cpu().numpy()
        elif cate==5:
          feature_map_lamp[int(xmin/8):int(xmax/8),int(ymin/8):int(ymax/8),:]=image_features.detach().cpu().numpy()
          

        
        gaussian = np.zeros((7,7))
        gaussian[3,3]=1
        gaussian=cv2.GaussianBlur(gaussian, (7,7), 0)
        topvalue=np.amax(gaussian)
        gaussian=gaussian*255/topvalue
        
        #semantic=
        #print (np.unique(semantic),np.unique(height),np.unique(orientation), np.unique(gaussian), 'gaussian',gaussian.shape)        
        #print ('xy', xmin, xmax, ymin,ymax)
        
        
        if cate!=5:
          instance[xmin:xmax,ymin:ymax]=cv2.resize(gaussian,(ymax-ymin,xmax-xmin))
        else:
          instance_lamp[xmin:xmax,ymin:ymax]=cv2.resize(gaussian,(ymax-ymin,xmax-xmin))
        
        '''cv2.imwrite(str(idx)+'semantic.png',semantic)
        cv2.imwrite(str(idx)+'height.png',height)
        #print ('height')
        cv2.imwrite(str(idx)+'orientation.png',orientation)
        #print ('ori')
        cv2.imwrite(str(idx)+'instance.png',instance)
        #print ('ins')
        #exit()'''


    return paths
    return renderables, [semantic,height,height2,orientation,instance,instance_lamp,feature_map,feature_map_lamp], object_count


def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def export_scene(output_directory, trimesh_meshes,  names=None):
    if names is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(trimesh_meshes))
    ]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(
            m,
            return_texture=True
        )

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def print_predicted_labels(dataset, boxes):
    object_types = np.array(dataset.object_types)
    box_id = boxes["class_labels"][0, 1:-1].argmax(-1)
    labels = object_types[box_id.cpu().numpy()].tolist()
    print("The predicted scene contains {}".format(labels))


def poll_specific_class(dataset):
    label = input(
        "Select an object class from {}\n".format(dataset.object_types)
    )
    if label in dataset.object_types:
        return dataset.object_types.index(label)
    else:
        return None


def make_network_input(current_boxes, indices):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices])
    )


def render_to_folder(
    args,
    folder,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    bbox_params,
    add_start_end=False
):
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu()

    if add_start_end:
        bbox_params_t = torch.cat([
            torch.zeros(1, 1, bbox_params_t.shape[2]),
            bbox_params_t,
            torch.zeros(1, 1, bbox_params_t.shape[2]),
        ], dim=1)

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels)
    )
    trimesh_meshes += tr_floor

    path_to_objs = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_objs):
        os.mkdir(path_to_objs)
    export_scene(path_to_objs, trimesh_meshes)

    path_to_image = os.path.join(
        args.output_directory,
        folder + "_render.png"
    )
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image, 1)
    ]
    render_simple_3dviz(
        renderables + floor_plan,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )


def render_scene_from_bbox_params(
    args,
    bbox_params,
    dataset,
    objects_dataset,
    classes,
    floor_plan,
    tr_floor,
    scene,
    path_to_image,
    path_to_objs
):
    boxes = dataset.post_process(bbox_params)
    print_predicted_labels(dataset, boxes)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu().numpy()

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t, objects_dataset, classes
    ) 
    renderables += floor_plan
    trimesh_meshes += tr_floor

    # Do the rendering
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image+".png", 1)
    ]
    render_simple_3dviz(
        renderables,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )
    if trimesh_meshes is not None:
        # Create a trimesh scene and export it
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)
        export_scene(path_to_objs, trimesh_meshes)
