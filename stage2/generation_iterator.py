import os
import trimesh
from data_processing.evaluation import eval_mesh
import traceback
import pickle as pkl
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp
from multiprocessing import Pool
import mcubes
import torch

# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p):

    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #print(out_path)


    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    #data_tupels = []
    
    dic=np.load('../data/official_chair_test.npy',allow_pickle=1)[()]
    #dic=np.load('../blip_balance100.npy',allow_pickle=1)[()]  
    #dic=np.load('../jingyudata/3dfront_test.npy',allow_pickle=1)[()]
 
    #dic=np.load('../blip_balance100.npy',allow_pickle=1)[()]
    #print (dic.keys())
    
    for i, data in tqdm(enumerate(loader)):


        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])
        
        
        name=path.split('/')[-1].split('_')[0].split('.')[0]
        self_text=path.split('/')[-1].split('_')[1]#.split('.')[0] 
        #print ('name', name)   
        
        if name not in dic.keys():
          continue
        texts=dic[name]
        
        #texts=['red red chair ']
        
        
        print (i)
        for text in texts:
              if text[:15]==self_text[:15]:
                  #try: #
                  #text='A red swivel chair with black wheels.'
                          
                  noise=torch.randn(1,128,1,1,1).cuda()*1 #diversified generation: 10
                  mesh,vertices, triangles= gen.generate_mesh(data,text,noise=noise)
                  namefull=path.split('/')[-1]
                  name=path.split('/')[-1].split('_')[0]
                  
                  
                  #trimesh.exchange.export.export_mesh(mesh, 'mesh_new/'+name+'_'+text[:50].replace('/','')+'.obj')
                  #print (text,'save')
                  #trimesh.exchange.export.export_mesh(mesh, 'mani0520/'+name+'_'+text[:50].replace('/','')+'.obj')
                  trimesh.exchange.export.export_mesh(mesh, 'mesh/'+name+self_text+text[:40].replace('/','')+str(i)+'.obj') #'mani2/'+name+'_'+text[:50].replace('/','')+'.obj')
                  #mcubes.export_obj(vertices, triangles, 'mani_new/'+self_text+'.obj')
                  #except:
                  #pass
      
                  
                  #data_tupels.append((logits,data, out_path))
    
      
              #except Exception as err:
              #    print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))
        
          


def save_mesh(data_tupel):
    logits, data, out_path = data_tupel
    mesh = gen.mesh_from_logits(logits)

    path = os.path.normpath(data['path'][0])
    export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    mesh.export(export_path + 'surface_reconstruction.off')

def create_meshes(data_tupels):
    p = Pool(mp.cpu_count())
    p.map(save_mesh, data_tupels)
    p.close()