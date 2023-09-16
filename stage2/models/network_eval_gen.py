import os
import importlib
import torch
import numpy as np
import mcubes
import torch.nn.functional as F
from data.data import SDFSamples
from models.network import MultiScaleMLP,SparseComposer, create_coordinates
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, SpacedDiffusion, space_timesteps
from utils.debugger import MyDebugger
import time
def process_state_dict(network_state_dict):
    for key, item in list(network_state_dict.items()):
        if 'module.' in key:
            new_key = key.replace('module.', '')
            network_state_dict[new_key] = item
            del network_state_dict[key]

    return network_state_dict




### Setting
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-03-31_19-42-28_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-06_17-33-46_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-08_13-20-54_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-11_18-07-01_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-13_14-01-48_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-19_11-24-26_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-22_19-45-58_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_00-07-02_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_00-22-05_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_12-16-46_Wavelet-Training-experiment'
# testing_folder = r'E:\proj51_debug\2022-04-26_15-11-28_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_15-17-20_Wavelet-Training-experiment'
# testing_folder = r'E:\proj51_debug\2022-04-26_20-53-49_Wavelet-Training-experiment'
# testing_folder = r'E:\proj51_debug\2022-04-26_21-02-43_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_20-54-07_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-26_20-55-17_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-27_12-25-56_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-27_21-25-48_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-28_17-46-15_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-27_21-25-48_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-29_13-38-55_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-29_19-47-03_Wavelet-Training-experiment'
# testing_folder = r'E:\testing_folder\exp_6' # Chair
# testing_folder = r'E:\testing_folder\exp_7'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-04_13-59-26_Wavelet-Training-experiment' # Chair
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-07_17-40-22_Wavelet-Training-experiment' # Chair 2
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-10_01-52-15_Wavelet-Training-experiment' # chair 3
# testing_folder = r'E:\testing_folder\2022-05-10_14-12-39_Wavelet-Training-experiment' # chair new batch 16
# testing_folder = r'E:\testing_folder\2022-05-11_16-43-53_Wavelet-Training-experiment' # chair new batch 64
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-12_15-20-17_Wavelet-Training-experiment'
testing_folder = r'E:\testing_folder\2022-05-13_01-53-53_Wavelet-Training-experiment' # Chair
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-04_13-59-26_Wavelet-Training-experiment'
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-05_15-30-07_Wavelet-Training-experiment' # Table
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-08_12-00-40_Wavelet-Training-experiment' # Table 2
# testing_folder = r'E:\testing_folder\exp_17' # Table
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-14_00-34-41_Wavelet-Training-experiment' # Cabinet
# testing_folder = r'E:\testing_folder\exp_9' # Cabinet
# testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-14_00-38-02_Wavelet-Training-experiment' # airplane

data_len = 5421 # chair
# data_len = 6807 # table
# data_len = 1256 # cabinet
# data_len = 3234 # airplane

data_path = r'Y:\data_per_category\03001627_chair\03001627_vox256_img_train.txt'
data_folder = r'Y:\sdf_samples_scaled_0.98\03001627'

config_path = os.path.join(testing_folder, 'config.py')

## import config here
spec = importlib.util.spec_from_file_location('*', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

### debugger
from configs import config as current_config
model_type = f"Wavelet-Decoding"


# epoch = 145 # table
# epoch = 500 # OLD CHAIR
# epoch = 450 # CHAIR
# epoch = 600 #225 #165 # chair 2
epoch = 2160 # chair 3
# epoch = 620 # table
# epoch = 5000
# epoch = 4080 # airplane
stage = 3
config.batch_num_points = 32 ** 3
use_preload = False
#loading_files = [(r'E:\03001627_3.npy', 3)]
# loading_files = [(r'E:\03001627_3.npy', 3), (r'E:\03001627_2.npy', 2)]
loading_files = [(r'E:\03001627_0.1_bior6.8_3.npy', 3), (r'E:\03001627_0.1_bior6.8_2.npy', 2)]

###
testing_cnt = 201
extra_start = 0
test_index = [i for i in range(config.max_depth + 1) if i not in config.zero_stages and i not in config.gt_stages]
assert len(test_index) == 1
test_index = test_index[0]
clip_noise = False
evaluate_gt = False
fixed_noise = False
need_gt = False
use_ddim = True
save_no_high = True
ddim_eta = 1.0
respacing = [config.diffusion_step // 10]
noise_path = r'E:\proj51_debug\2022-05-17_12-42-13_Network-Marching-Cubes-Diffusion-Gen\229_265_noise.pt' #

#
use_high_level_network = True

if use_high_level_network:
    high_level_folder = r'E:\testing_folder\2022-05-13_01-53-53_Wavelet-Training-experiment\high_levels'
    high_level_config_path = os.path.join(high_level_folder, 'config.py')

    spec = importlib.util.spec_from_file_location('*', high_level_config_path)
    high_level_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(high_level_config)



    high_level_stage = 0
    high_level_epoch = 80
    high_test_index = [i for i in range(high_level_config.max_depth + 1) if i not in high_level_config.zero_stages and i not in high_level_config.gt_stages]
    assert len(high_test_index) == 1
    high_test_index = high_test_index[0]
    high_level_network_path =  os.path.join(high_level_folder, f'model_epoch_{high_level_stage}_{high_level_epoch}.pth')

else:
    high_level_config = None

def one_generation_process(args):

    cuda_id, start_index, testing_cnt, folder_path = args
    device = torch.device(f'cuda:{cuda_id}')
    network_path = os.path.join(testing_folder, f'model_epoch_{stage}_{epoch}.pth')

    ### create dataset
    samples = SDFSamples(data_path=data_path,
                         data_folder=data_folder,
                         resolution=config.resolution,
                         num_points=config.num_points,
                         interval=config.interval,
                         first_k=config.first_k,
                         use_surface_samples=False,
                         data_files=loading_files,
                         use_preload=use_preload
                         )
    ### level_indices_remap
    level_map = {idx: level for idx, (_, level) in enumerate(loading_files)}

    samples.return_all = True

    with torch.no_grad():
        ### initialize network
        dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
            device)
        dwt_forward_3d_lap = DWTForward3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
            device)
        composer_parms = dwt_inverse_3d_lap if config.use_dense_conv else None
        dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution],
                                             J=config.max_depth,
                                             wave=config.wavelet, mode=config.padding_mode,
                                             inverse_dwt_module=composer_parms).to(
            device)
        network = MultiScaleMLP(config=config, data_num=data_len, J=config.max_depth,
                                shape_list=dwt_sparse_composer.shape_list)

        network_state_dict = torch.load(network_path,map_location=f'cuda:{cuda_id}')
        network_state_dict = process_state_dict(network_state_dict)

        network.load_state_dict(network_state_dict)
        network = network.to(device)
        network.eval()

        if use_high_level_network:
            high_level_network = MultiScaleMLP(config=high_level_config, data_num=data_len, J=config.max_depth,
                                shape_list=dwt_sparse_composer.shape_list)
            high_level_network_state_dict = torch.load(high_level_network_path, map_location=f'cuda:{cuda_id}')
            high_level_network_state_dict = process_state_dict(high_level_network_state_dict)
            high_level_network.load_state_dict(high_level_network_state_dict)
            high_level_network = high_level_network.to(device)
            high_level_network.eval()
        else:
            high_level_network = None


        betas = get_named_beta_schedule(config.diffusion_beta_schedule, config.diffusion_step,
                                        config.diffusion_scale_ratio)

        diffusion_module = SpacedDiffusion(use_timesteps=space_timesteps(config.diffusion_step, respacing),
                                           betas=betas,
                                           model_var_type=config.diffusion_model_var_type,
                                           model_mean_type=config.diffusion_model_mean_type,
                                           loss_type=config.diffusion_loss_type)

        testing_indices = [265] * testing_cnt
        if fixed_noise:
            if noise_path is not None:
                noise = torch.load(noise_path, map_location=f'cuda:{cuda_id}').to(device)
            else:
                noise = torch.randn([1, 1] + dwt_sparse_composer.shape_list[test_index]).to(device)
        else:
            noise = None


        for m in range(testing_cnt):
            testing_sample_index = testing_indices[m]
            if use_preload:
                data = samples[testing_sample_index][0]
                low_lap, highs_lap = None, [None] * config.max_depth
                coeff_gt = data
                for j, gt in enumerate(coeff_gt):
                    level = level_map[j]
                    if level == config.max_depth:
                        low_lap = torch.from_numpy(coeff_gt[j]).unsqueeze(0).unsqueeze(1).to(device)
                    else:
                        highs_lap[level] = torch.from_numpy(coeff_gt[j]).unsqueeze(0).unsqueeze(1).to(device)
            else:
                voxels_np = np.load(samples.data_path[testing_sample_index])
                voxels_cuda = torch.from_numpy(voxels_np).float().to(device).unsqueeze(0).unsqueeze(0)
                low_lap, highs_lap = dwt_forward_3d_lap(voxels_cuda)

            low_lap = torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[config.max_depth])).float().to(
                device) if low_lap is None else low_lap
            highs_lap = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[j])).float().to(device) if highs_lap[
                                                                                                                  j] is None else
                         highs_lap[j] for j in range(config.max_depth)]

            if need_gt:
                voxels_pred = dwt_inverse_3d_lap((low_lap, highs_lap))
                vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
                vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
                mcubes.export_off(vertices, traingles, os.path.join(folder_path, f'{testing_sample_index}_gt.off'))

            if not evaluate_gt:
                if test_index == config.max_depth:
                    model_kwargs = {'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt')}
                    if use_ddim:
                        low_samples = diffusion_module.ddim_sample_loop(model=network.low_layer,
                                                                        shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                                        device=device,
                                                                        clip_denoised=clip_noise, progress=True,
                                                                        noise=noise,
                                                                        eta=ddim_eta,
                                                                        model_kwargs=model_kwargs).detach()
                    else:
                        low_samples = diffusion_module.p_sample_loop(model=network.low_layer,
                                                                     shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                                     device=device,
                                                                     clip_denoised=clip_noise, progress=True, noise=noise,
                                                                     model_kwargs=model_kwargs).detach()

                    highs_samples = [torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[i]), device=device) for i in
                                     range(config.max_depth)]

                    if use_high_level_network:
                        highs_samples[high_test_index] = high_level_network.extract_full_coeff(code_indices=None, level=high_test_index, stage=stage,
                                                           zero_stages=high_level_config.zero_stages,
                                                           gt_stages=high_level_config.gt_stages, gt_low=low_samples,
                                                           gt_highs=None).detach()

                else:
                    low_samples = low_lap
                    highs_samples = []
                    for i in range(config.max_depth):
                        if i < test_index:
                            highs_samples.append(
                                torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[i]), device=device))
                        elif i == test_index:
                            low_cuda_lap_upsampled = F.interpolate(low_lap, highs_lap[i].size()[-3:])
                            model_kwargs = {'low_cond': low_cuda_lap_upsampled,
                                            'noise_save_path': os.path.join(folder_path, f'{m+start_index}_{testing_sample_index}_noise.pt')}
                            if use_ddim:
                                sample = diffusion_module.ddim_sample_loop(model=network.low_layer,
                                                                           shape=[1, 1] +
                                                                                 dwt_sparse_composer.shape_list[-1],
                                                                           device=device,
                                                                           clip_denoised=clip_noise, progress=True,
                                                                           noise=noise,
                                                                           eta=ddim_eta,
                                                                           model_kwargs=model_kwargs).detach()
                            else:
                                sample = diffusion_module.p_sample_loop(model=network.highs_layers[i],
                                                                        shape=[1, 1] + dwt_sparse_composer.shape_list[i],
                                                                        device=device,
                                                                        clip_denoised=clip_noise, progress=True,
                                                                        model_kwargs=model_kwargs, noise=noise).detach()
                            highs_samples.append(sample)
                        else:
                            highs_samples.append(highs_lap[i])

                voxels_pred = dwt_inverse_3d_lap((low_samples, highs_samples))
                vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
                vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
                mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'{m+start_index+extra_start}_{testing_sample_index}.obj'))

                if save_no_high:
                    highs_samples[high_test_index] = torch.zeros_like(highs_samples[high_test_index]).to(device)
                    voxels_pred = dwt_inverse_3d_lap((low_samples, highs_samples))
                    vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
                    vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
                    mcubes.export_obj(vertices, traingles, os.path.join(folder_path,
                                                                        f'{m + start_index + extra_start}_{testing_sample_index}_no_highs.obj'))

                print(f"Done {os.path.join(folder_path,f'{m+start_index+extra_start}_{testing_sample_index}.off')}!")

if __name__ == '__main__':

    debugger = MyDebugger(f'Network-Marching-Cubes-Diffusion-Gen',
                          is_save_print_to_file=False)

    from torch.multiprocessing import Pool

    GPU_CNT = 1
    PER_GPU_PROCESS = 1
    pool = Pool(GPU_CNT * PER_GPU_PROCESS)


    args = []
    assert testing_cnt % (GPU_CNT * PER_GPU_PROCESS) == 0
    if GPU_CNT * PER_GPU_PROCESS > 1:
        per_process_data_num = testing_cnt // (GPU_CNT * PER_GPU_PROCESS)
        for i in range(GPU_CNT):
            for j in range(PER_GPU_PROCESS):
                args.append((i, (i * PER_GPU_PROCESS + j) * per_process_data_num, per_process_data_num, debugger.file_path('.')))

        pool.map(one_generation_process, args)
    else:
        one_generation_process((0, 0, testing_cnt,  debugger.file_path('.')))

    print("done!")

