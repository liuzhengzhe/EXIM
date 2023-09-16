import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback
from pysdf import SDF
ROOT = '../data/shapenet/data'
import torch
import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import block_reduce
from libvoxelize.voxelize import voxelize_mesh_
from libmesh.inside_mesh import check_mesh_contains
import mcubes
# From Occupancy Networks, Mescheder et. al. CVPR'19

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def boundary_sampling(path):
    try:

        #if os.path.exists(path +'/boundary_{}_samples.npz'.format(args.sigma)):
        #    return

        off_path = path + '/model_flipped_lzz_scaled.obj.off'
        out_file = path +'/sdf_{}_samples.npz'.format(args.sigma)

        mesh = trimesh.load(off_path)
        
        f = SDF(mesh.vertices, mesh.faces)
        
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        #occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]
        
        
        occupancies = f(boundary_points)
        


        
        '''resolution=256

        shape = (resolution,) * 3
        bb_min = (0.5,) * 3
        bb_max = (resolution - 0.5,) * 3
        # Create points. Add noise to break symmetry
        points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
        #points = points + 0.1 * (np.random.rand(*points.shape) - 0.5)
        points = (points / 256 - 0.5)
        #print (points)
        occ = f(points)
        occ=np.reshape(occ,(256,256,256))
        #print (occ[0,0,0], occ[127,127,127])
        
        for i in range(100,200):
          print (i,-occ[i,i,i])
        #print (np.unique(points), points.shape, occ.shape)
        #exit()'''
        
        
                          
        #vertices, traingles = mcubes.marching_cubes(occ, 0.0)
        #vertices = (vertices.astype(np.float32) - 0.5) /  256 - 0.5
        #mcubes.export_obj(vertices, traingles,  'cube.obj')        


        '''occ = f([0,0,0])
        occ = f([-0.5,-0.5,-0.5])
        
        origin_contained = f.contains([0, 0, 0])
        print (origin_contained)
        origin_contained = f.contains([-0.5,-0.5,-0.5])
        print (origin_contained)
        
        print (np.unique(boundary_points))'''
        
        #print (np.unique(boundary_points))
        
        occupancies*=(-2)
        
        occupancies[np.where(occupancies>0.1)]=0.1
        occupancies[np.where(occupancies<-0.1)]=-0.1
        
        #print ('boundary_points', np.unique(occupancies), occupancies.shape, boundary_points.shape)
        
        
        #print (boundary_points.shape, occupancies.shape, grid_coords.shape)

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float)

    args = parser.parse_args()


    sample_num = 100000


    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, glob.glob( ROOT + '/*/*/')) #'/mnt/sda/lzz/metric_jingyu/03001627/1028b32dc1873c2afe26a3ac360dbd4/')) #ROOT + '/*/*/'))
