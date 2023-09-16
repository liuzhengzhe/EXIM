import open3d as o3d
import glob
from multiprocessing import Pool
import numpy as np
paths=glob.glob('shapenet/data/03001627/*/model_flipped_manifold_*.obj')
for path in paths:
  #def process(path):
  name=path.split('/')[-2]
  mesh = o3d.io.read_triangle_mesh(path)
  print (path)
  
  R = mesh.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
  mesh.rotate(R, center=(0, 0, 0))
  
  o3d.io.write_triangle_mesh('shapenet/data/03001627/'+name+'/model_flipped_lzz.obj', mesh)



#num_thread=40
#with Pool(num_thread) as p:
#    p.map(process, paths)

  
  