import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy.linalg import sqrtm
#from scipy.misc import imread
#import imageio.imread as imread
from torch.nn.functional import adaptive_avg_pool2d
#import sys
#sys.path.insert(1, './evaluation')
from pointnet import PointNetCls
from dataset_benchmark import BenchmarkDataset

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

"""Calculate Frechet Pointcloud Distance referened by Frechet Inception Distance."
    [ref] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
    github code  : (https://github.com/bioinf-jku/TTUR)
    paper        : (https://arxiv.org/abs/1706.08500)

"""

def get_activations(pointclouds, model, batch_size=100, dims=1808,
                    device=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    n_batches = pointclouds.size(0) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    pointclouds = pointclouds.transpose(1,2)
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        pointcloud_batch = pointclouds[start:end]
        
        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)

        _, _, actv = model(pointcloud_batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = actv.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(pointclouds, model, batch_size=100,
                                    dims=1808, device=None, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(pointclouds, model, batch_size, dims, device, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['m'][:], f['s'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s

def save_statistics(real_pointclouds, path, model, batch_size, dims, cuda):
    m, s = calculate_activation_statistics(real_pointclouds, model, batch_size,
                                         dims, cuda)
    np.savez(path, m = m, s = s)
    print('save done !!!')

def calculate_fpd(pointclouds1, pointclouds2=None, batch_size=100, dims=1808, device=None):
    """Calculates the FPD of two pointclouds"""

    PointNet_path = './evaluation/cls_model_39.pth'
    statistic_save_path = './evaluation/pre_statistics.npz'
    model = PointNetCls(k=16)
    model.load_state_dict(torch.load(PointNet_path))
    
    if device is not None:
        model.to(device)

    m1, s1 = calculate_activation_statistics(pointclouds1, model, batch_size, dims, device)
    if pointclouds2 is not None:
        m2, s2 = calculate_activation_statistics(pointclouds2, model, batch_size, dims, device)
    else: # Load saved statistics of real pointclouds.
        f = np.load(statistic_save_path)
        m2, s2 = f['m'][:], f['s'][:]
        f.close()
        
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print ('result', fid_value)
    return fid_value




import numpy as np

f=np.load('./evaluation/pre_statistics.npz')
m2, s2 = f['m'][:], f['s'][:]
#print (m2.shape,s2.shape)
#exit()

import glob
paths=glob.glob('gt/*.pts') #it is from: pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points
points=[]
for path in paths[1:]:
  name=path.split('/')[-1]
  point_set=np.loadtxt(path)

  point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center

  dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
  point_set = point_set / dist #scale

  idx = np.random.randint(point_set.shape[0], size=2048)

  point_set=point_set[idx,:]

  points.append(point_set)





gt_s=torch.from_numpy(np.asarray(points)).float() #.cuda()


import numpy as np
import glob

import open3d as o3d

a_s=[]
dic={}

paths=glob.glob('pred/*.ply')
for path in paths:
  name=path.split('/')[-1].split('_')[0]

  dic[name]=1
  pcd = o3d.io.read_point_cloud(path)
  pcd=np.asarray(pcd.points)
  if pcd.shape[0]==0:
    continue

  pcd=(pcd+0.5)*8+0.5
  pcd=(pcd-0.5)/64-0.5

  a=np.zeros(pcd.shape)
  a[:,0]=pcd[:,2]
  a[:,1]=pcd[:,1]
  a[:,2]=pcd[:,0]
  a[:,0]=-a[:,0]
  
  
  point_set=a


  point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
  dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)

  point_set = point_set / dist #scale
  if dist==0:
    continue

  idx = np.random.randint(point_set.shape[0], size=2048)
  point_set=point_set[idx,:]

  if np.isnan(point_set).any():
    continue
  whe=np.where(point_set!=point_set)[0]

  if whe.shape[0]!=0:
    print (whe.shape[0], 'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
    continue
  a_s.append(point_set*0.92)

  '''from plyfile import PlyData,PlyElement
  some_array=[]
  size=258
  for i in range(a.shape[0]):
       some_array.append((a[i,0],a[i,1],a[i,2]))
  some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32')])
  el = PlyElement.describe(some_array, 'vertex')
  PlyData([el]).write('pred/'+name+'.ply')'''


a_s=torch.from_numpy(np.asarray(a_s)).float() #.cuda()
calculate_fpd(gt_s,a_s)