

import torch
shape=torch.load('/mnt/sda/lzz/ifnet/experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/checkpoints//checkpoint_epoch_914.tar')
color=torch.load('experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/checkpoints/checkpoint_epoch_1490.tar')




for k in shape['model_state_dict'].keys():
  print ('shape', k,shape['model_state_dict'][k].shape)
  #color['model_state_dict'][k]=shape['model_state_dict'][k]#[:]

for k in color['model_state_dict'].keys():
  print ('color',k,color['model_state_dict'][k].shape)
  #color['model_state_dict'][k]=shape['model_state_dict'][k]#[:]


torch.save(color,'experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v128_mShapeNet128Vox/checkpoints/checkpoint_epoch_1491.tar')