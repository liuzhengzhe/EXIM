from utils.debugger import MyDebugger
from configs import config
import torch
import numpy as np
from models.module.dwt import DWTInverse3d_Laplacian
from models.network import SparseComposer
import mcubes

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if __name__ == '__main__':
    debugger = MyDebugger('My-Testing Coefficient', is_save_print_to_file = False)

    level_2_path = r'Y:\sdf_samples_scaled\03001627_0.1_bior6.8_2_zero.npy'
    level_3_path =r'Y:\sdf_samples_scaled\03001627_0.1_bior6.8_3_zero.npy'

    dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    dwt_sparse_composer = SparseComposer(input_shape = [config.resolution, config.resolution, config.resolution], J = config.max_depth, wave=config.wavelet, mode = config.padding_mode, inverse_dwt_module=None).to(device)

    level_2_coefficient = np.load(level_2_path)
    level_3_coefficient = np.load(level_2_path)

    test_cnt = 100
    for i in range(test_cnt):
        highs_level_coefficients = [torch.zeros([1, 1] + dwt_sparse_composer.shape_list[j]) for j in range(config.max_depth)]
        low_level_coefficients = torch.from_numpy(level_3_coefficient[-i]).unsqueeze(0).unsqueeze(0).to(device)
        highs_level_coefficients[-1] = torch.from_numpy(level_2_coefficient[-i]).unsqueeze(0).unsqueeze(0).to(device)

        ### reconstruction
        voxels_reconstruction = dwt_inverse_3d_lap((low_level_coefficients, highs_level_coefficients))

        vertices, traingles = mcubes.marching_cubes(voxels_reconstruction.detach().cpu().numpy()[0, 0], 0.0)
        vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
        mcubes.export_obj(vertices, traingles, debugger.file_path(
            f'{i}.obj'))
        print(f"{i} done!")