
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

def visualization(folder_path, config, network, sample_indices, dwt_sparse_composer, dwt_inverse_3d_lap):

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    test_index = [i for i in range(config.max_depth + 1) if i not in config.zero_stages and i not in config.gt_stages]
    assert len(test_index) == 1 and test_index[0] == config.max_depth
    test_index = test_index[0]

    for sample_index in sample_indices:
        start_time = time.time()
        code_indices = torch.from_numpy(np.array([sample_index])).to(device).long()

        low_coeff = network.extract_full_coeff(code_indices = code_indices, level= test_index, stage=0)
        highs_coeff = [torch.zeros([1,1]+list(dwt_sparse_composer.shape_list[i])).to(device) for i in range(config.max_depth)]

        reconstructions = dwt_inverse_3d_lap((low_coeff, highs_coeff))
        vertices, traingles = mcubes.marching_cubes(reconstructions.detach().cpu().numpy()[0, 0], 0.0)
        mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'sample_{sample_index}.obj'))



        print(f"index {sample_index} done in {time.time() - start_time}")

def visualization_random_sample(folder_path, config, network, sample_indices, dwt_sparse_composer, dwt_inverse_3d_lap):

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    test_index = [i for i in range(config.max_depth + 1) if i not in config.zero_stages and i not in config.gt_stages]
    assert len(test_index) == 1 and test_index[0] == config.max_depth
    test_index = test_index[0]

    for sample_index in sample_indices:

        start_time = time.time()
        code_indices = torch.from_numpy(np.array([sample_index])).to(device).long()
        codes = torch.randn((1, config.latent_dim)).to(device)

        low_coeff = network.extract_full_coeff(code_indices = code_indices, level= test_index, stage=0, VAD_codes=codes)
        highs_coeff = [torch.zeros([1,1]+list(dwt_sparse_composer.shape_list[i])).to(device) for i in range(config.max_depth)]

        reconstructions = dwt_inverse_3d_lap((low_coeff, highs_coeff))
        vertices, traingles = mcubes.marching_cubes(reconstructions.detach().cpu().numpy()[0, 0], 0.0)
        mcubes.export_obj(vertices, traingles, os.path.join(folder_path, f'random_{sample_index}.obj'))

        print(f"index {sample_index} done in {time.time() - start_time}")

if __name__ == '__main__':
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-03-31_19-42-28_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-06_17-33-46_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-08_13-20-54_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-11_18-07-01_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-13_14-01-48_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-19_11-24-26_Wavelet-Training-experiment'
    # testing_folder = r'Y:\ImplicitWavelet\debug\2022-04-22_19-45-58_Wavelet-Training-experiment'
    testing_folder = r'Y:\ImplicitWavelet\debug\2022-05-16_22-59-50_Wavelet-Training-experiment'

    #
    data_path = r'Y:\data_per_category\03001627_chair\03001627_vox256_img_train.txt'
    data_folder = r'Y:\sdf_samples\03001627'

    config_path = os.path.join(testing_folder, 'config.py')

    ## import config here
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ### debugger
    model_type = f"VAD"
    debugger = MyDebugger(f'Network-Marching-Cubes-SDF-{model_type}',
                          is_save_print_to_file=False)

    ###
    epoch = 2180
    stage = 3
    config.batch_num_points = 32 ** 3

    network_path = os.path.join(testing_folder, f'model_epoch_{stage}_{epoch}.pth')

    ### create dataset
    # samples = SDFSamples(data_path=data_path,
    #                      data_folder=data_folder,
    #                      resolution=config.resolution,
    #                      num_points=config.num_points,
    #                      interval=config.interval,
    #                      first_k=config.first_k,
    #                      use_surface_samples=False
    #                      )
    data_len = 5421

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
    network = MultiScaleMLP(config=config, data_num=data_len, J=config.max_depth, shape_list = dwt_sparse_composer.shape_list)

    network_state_dict = torch.load(network_path)
    network_state_dict = process_state_dict(network_state_dict)

    network.load_state_dict(network_state_dict)
    network = network.to(device)

    sample_indices = range(20)


    visualization(folder_path = debugger.file_path('.'),
                  config = config,
                  network = network,
                  sample_indices = sample_indices,
                  dwt_sparse_composer = dwt_sparse_composer,
                  dwt_inverse_3d_lap=dwt_inverse_3d_lap)

    visualization_random_sample(folder_path = debugger.file_path('.'),
                  config = config,
                  network = network,
                  sample_indices = sample_indices,
                  dwt_sparse_composer = dwt_sparse_composer,
                  dwt_inverse_3d_lap=dwt_inverse_3d_lap)