import os
import importlib
import torch
import numpy as np
import mcubes
from data.data import SDFSamples
from models.network import MultiScaleMLP,SparseComposer, create_coordinates
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from utils.debugger import MyDebugger
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def process_state_dict(network_state_dict):
    for key, item in list(network_state_dict.items()):
        if 'module.' in key:
            new_key = key.replace('module.', '')
            network_state_dict[new_key] = item
            del network_state_dict[key]

    return network_state_dict

def visualization(folder_path, config, network, sample_indices, all_indices, dwt_sparse_composer, dwt_forward_3d_lap, dwt_inverse_3d_lap, samples):

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    for sample_index in sample_indices:
        start_time = time.time()
        code_indices = torch.from_numpy(np.array([sample_index])).to(device).long()
        voxels_reconstruction = []
        batch_num = config.batch_num_points // config.num_points
        voxels_np = np.load(samples.data_path[sample_index])
        voxels_cuda = torch.from_numpy(voxels_np).float().to(device).unsqueeze(0).unsqueeze(0)
        low_cuda_lap, highs_cuda_lap = dwt_forward_3d_lap(voxels_cuda)



        ### compute the gt
        args_dict = {'code_indices': code_indices,
                     'stage': config.max_depth,
                     'zero_stages': config.zero_stages if hasattr(config, 'zero_stages') else [],
                     'gt_stages': config.gt_stages if hasattr(config, 'gt_stages') else [],
                     'save_high': True}
        if len(args_dict.get('gt_stages')) > 0:
            args_dict['gt_low'] = low_cuda_lap
            args_dict['gt_highs'] = highs_cuda_lap

            new_highs_cuda_lap = [None] * len(highs_cuda_lap)
            for j in range(config.max_depth):
                if j not in config.gt_stages:
                    new_highs_cuda_lap[j] = torch.zeros_like(highs_cuda_lap[j]).to(device)
                else:
                    new_highs_cuda_lap[j] = highs_cuda_lap[j]

            voxels_reconstruction_lap = dwt_inverse_3d_lap((low_cuda_lap, new_highs_cuda_lap))
            vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap.detach().cpu().numpy()[0, 0], 0.0)
            mcubes.export_off(vertices, traingles, os.path.join(folder_path, f'{sample_index}_only_gt.off'))

        for j in range(all_indices.size(0) // batch_num):
            voxels_pred = dwt_sparse_composer(all_indices[j * batch_num:(j + 1) * batch_num].view(1, -1, 3),
                                              network, **args_dict).detach()
            voxels_reconstruction.append(voxels_pred)
        voxels_reconstruction = torch.cat(voxels_reconstruction, dim=0)
        voxels_reconstruction = voxels_reconstruction.view((1, 1, config.resolution, config.resolution, config.resolution))
        voxels_pred = voxels_reconstruction.detach().cpu().numpy()[0, 0]
        voxels_pred = np.swapaxes(voxels_pred, 0, 1)
        vertices, traingles = mcubes.marching_cubes(voxels_pred, 0.0)
        mcubes.export_off(vertices, traingles,
                          os.path.join(folder_path, f'{sample_index}.off'))




        for i in range(stage):
            highs_cuda_lap[i] = torch.zeros_like(highs_cuda_lap[i]).to(device)
        voxels_reconstruction_lap = dwt_inverse_3d_lap((low_cuda_lap, highs_cuda_lap))

        vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap.detach().cpu().numpy()[0, 0], 0.0)
        mcubes.export_off(vertices, traingles, os.path.join(folder_path, f'{sample_index}_gt.off'))

        print(f"index {sample_index} done in {time.time() - start_time}")

if __name__ == '__main__':
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-03-31_19-42-28_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-06_17-33-46_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-08_13-20-54_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-11_18-07-01_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-13_14-01-48_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-19_11-24-26_Wavelet-Training-experiment'
    testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-22_19-45-58_Wavelet-Training-experiment'

    #
    data_path = r'Y:\data_per_category\03001627_chair\03001627_vox256_img_train.txt'
    data_folder = r'Y:\sdf_samples\03001627'

    config_path = os.path.join(testing_folder, 'config.py')

    ## import config here
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ### debugger
    model_type = f"Wavelet-Decoding"
    debugger = MyDebugger(f'Network-Marching-Cubes-SDF-{model_type}',
                          is_save_print_to_file=False)

    ###
    epoch = 10
    stage = 0
    config.batch_num_points = 32 ** 3

    network_path = os.path.join(testing_folder, f'model_epoch_{stage}_{epoch}.pth')

    ### create dataset
    samples = SDFSamples(data_path=data_path,
                         data_folder=data_folder,
                         resolution=config.resolution,
                         num_points=config.num_points,
                         interval=config.interval,
                         first_k=config.first_k,
                         use_surface_samples=False
                         )

    ### initialize network
    dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(
        device)
    dwt_forward_3d_lap = DWTForward3d_Laplacian(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    composer_parms = dwt_inverse_3d_lap if config.use_dense_conv else None
    dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution],
                                         J=config.max_depth,
                                         wave=config.wavelet, mode=config.padding_mode,
                                         inverse_dwt_module=composer_parms).to(
        device)
    network = MultiScaleMLP(config=config, data_num=len(samples), J=config.max_depth, shape_list = dwt_sparse_composer.shape_list)

    network_state_dict = torch.load(network_path)
    network_state_dict = process_state_dict(network_state_dict)

    network.load_state_dict(network_state_dict)
    network = network.to(device)

    #sample_indices = range(len(samples))
    sample_indices = range(20)

    all_indices = create_coordinates(config.resolution).view((-1, 3))
    all_indices = all_indices.view((-1, config.num_points, 3)).int().to(device)

    visualization(folder_path = debugger.file_path('.'),
                  config = config,
                  network = network,
                  sample_indices = sample_indices,
                  all_indices=all_indices,
                  dwt_sparse_composer = dwt_sparse_composer,
                  dwt_forward_3d_lap=dwt_forward_3d_lap,
                  dwt_inverse_3d_lap=dwt_inverse_3d_lap,
                  samples = samples)


    print("done!")

