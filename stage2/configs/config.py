import pywt
import torch


debug_base_folder = "../debug"
backup_path = 'backup'


data_path = '../datas/03001627_vox256_img_train.txt'
data_folder = '/home/zzliu/ShapeNetCore.v1/03001627/'
use_preload = True
data_files = [('03001627_0.1_bior6.8_3_zero.npy', 3), ('03001627_0.1_bior6.8_2_zero.npy',2)]
num_points =  4096
interval = 1
test_nums = 16
batch_num_points = 64 ** 3
first_k = None
use_surface_samples = False
sample_resolution = 64
sample_ratio = 0.8
load_ram = False
loss_function = torch.nn.MSELoss()
new_low_fix = True
remove_reductant = True
mix_precision = True

#
batch_size = 12
lr = 1e-4
lr_decay = False
lr_decay_feq = 500
lr_decay_rate = 0.998
vis_results = True
progressive = True
data_worker = 25
beta1 = 0.9
beta2 = 0.999
optimizer = torch.optim.Adam

## network
resolution = 256
latent_dim = 256
padding_mode = 'zero'
wavelet_type = 'bior6.8'
wavelet = pywt.Wavelet(wavelet_type)
max_depth = pywt.dwt_max_level(data_len = resolution, filter_len=wavelet.dec_len)
latent_dim = int(resolution // (max_depth+1) * (max_depth+1)) # round down
activation = torch.nn.LeakyReLU(0.02) #torch.nn.LeakyReLU(0.02)
use_fourier_features = True
fourier_norm = 1.0
use_dense_conv = True
linear_layers = [128, 128, 128, 128]
code_bound = 0.1
weight_sigma = 0.02
scale_coordinates = True
train_only_current_level = True
lr_weight_decay_after_stage = False
lr_decay_rate_after_stage = 0.01
use_clip = False
clip_half = False
clip_bound = 0.1
train_low_stage_with_full = False
use_gradient_clip = False
gradient_clip_value = 1.0
use_instance_norm = True
use_instance_affine = True
use_layer_norm = False
use_layer_affine = False
training_stage = 3
train_with_gt_coeff = True

### diffusion setting
from models.module.gaussian_diffusion import  ModelMeanType, ModelVarType, LossType
use_diffusion = True
diffusion_step = 1000
diffusion_model_var_type = ModelVarType.FIXED_SMALL
diffusion_learn_sigma = False
diffusion_sampler = 'second-order'
diffusion_model_mean_type = ModelMeanType.EPSILON
diffusion_rescale_timestep = False
diffusion_loss_type = LossType.MSE
diffusion_beta_schedule = 'linear'
diffusion_scale_ratio = 1.0
unet_model_channels = 64
unet_num_res_blocks = 3
unet_channel_mult = (1, 1, 2, 4)
unet_channel_mult_low = (1, 2, 2, 2)
unet_activation = None #torch.nn.LeakyReLU(0.1)
attention_resolutions = []
if diffusion_learn_sigma:
    diffusion_model_var_type = ModelVarType.LEARNED_RANGE
    diffusion_loss_type = LossType.RESCALED_MSE

###
highs_use_conv3d = False
conv3d_tuple_layers_highs_append = [
                       (8, (5, 5, 5), (2, 2, 2)),
                       (8, (5, 5, 5), (1, 1, 1)),
                       ]

###
highs_use_downsample_features = False
highs_use_unent = False
downsample_features_dim = 64
conv3d_downsample_tuple_layers = [
    (16, (3, 3, 3), (2, 2, 2)),
    (16, (3, 3, 3), (1, 1, 1)),
    (32, (3, 3, 3), (2, 2, 2)),
    (32, (3, 3, 3), (1, 1, 1)),
    (64, (3, 3, 3), (2, 2, 2)),
    (64, (3, 3, 3), (1, 1, 1)),
    (128, (3, 3, 3), (2, 2, 2)),
    (128, (3, 3, 3), (1, 1, 1)),
]

# use discriminator
use_discriminator = False
discriminator_weight = 0.01
i_dim = 2
z_dim = 1
d_dim = 16

## resume
starting_epoch = 1160
training_epochs = 300000
saving_intervals = 50
starting_stage = max_depth
special_symbol = ''
network_resume_path = 'debug/2023-01-03_15-52-31_Wavelet-Training-experiment//model_epoch_3_2000.pth'
optimizer_resume_path = None #'debug/2022-11-20_23-10-19_Wavelet-Training-experiment/optimizer_epoch_3_1160.pth' #'debug/2022-10-04_04-50-36_Wavelet-Training-experiment/optimizer_epoch_3_61.pth' #None #r'/root/autodl-tmp/ImplicitWavelet/debug/2022-05-11_16-43-53_Wavelet-Training-experiment/optimizer_epoch_3_930.pth'
discriminator_resume_path = None
discriminator_opt_resume_path = None
exp_idx = 15
