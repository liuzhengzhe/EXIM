from utils.debugger import MyDebugger
import os
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
import mcubes
import pywt
from configs import config
import torch

def get_voxels_from_path(path):
    print(f'start {path}!')
    if path.endswith('.npy'):
        save_path = path
    else:
        save_path = os.path.join(config.backup_path, os.path.basename(path) + f'_{resolution}.npy')
    if os.path.exists(save_path):
        voxels = np.load(save_path)
    else:
        start_time = time.time()
        if os.path.isdir(path):
            mesh = trimesh.load_mesh(os.path.join(path, 'model_flipped_manifold.obj'))
        else:
            mesh = trimesh.load_mesh(path)
        voxels = mesh_to_voxels(mesh, resolution)
        np.save(save_path, voxels)
        print(f"Compelete time : {time.time() - start_time} s!")
    print(f"saved path : {save_path}")

    return voxels

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    import time
    from models.network import create_coordinates
    from models.module.dwt import DWTForward3d_Laplacian, DWTInverse3d_Laplacian
    debugger = MyDebugger('My-Wavelet-Swapping'
                          '', is_save_print_to_file = False)
    folder = r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627'
    cnt = 100

    paths = [
     # (r'Y:\sdf_samples\03001627\1c5d66f3e7352cbf310af74324aae27f_256.npy', r'Y:\\sdf_samples\\03001627\\1006be65e7bc937e9141f9b58470d646_256.npy'),
        (r'Y:\sdf_samples\03001627\60328528e791d7281f47fd90378714ec_256.npy', r'Y:\sdf_samples\03001627\892381333dc86d0196a8a62cbb17df9_256.npy')
    ]



    ## TESTING
    padding_mode = 'zero'
    wavelet_type = 'bior6.8'
    resolution = 256
    num_points = 4096
    batch_num_points = 32**3
    wavelet = pywt.Wavelet(wavelet_type)
    max_depth = pywt.dwt_max_level(data_len=resolution, filter_len=wavelet.dec_len)
    # max_depth = 4
    # max_depth = 1
    swap_level = [0, 1, 2]
    skip_level = [0, 1]

    indices = create_coordinates(resolution).view((-1, 3))
    all_indices = indices
    all_indices_np = np.reshape(all_indices.cpu().numpy(), (resolution, resolution, resolution, 3))
    all_indices = all_indices.view((-1, num_points, 3)).int().to(device)

    dwt_forward_3d_lap = DWTForward3d_Laplacian(J = max_depth, wave=wavelet, mode=padding_mode).to(device)
    dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)

    for path_1, path_2 in paths:
        voxel_1 = get_voxels_from_path(path_1)
        voxel_2 = get_voxels_from_path(path_2)

        voxel_1_cuda = torch.from_numpy(voxel_1).to(device).unsqueeze(0).unsqueeze(0)
        voxel_2_cuda = torch.from_numpy(voxel_2).to(device).unsqueeze(0).unsqueeze(0)

        low_cuda_1_lap, highs_cuda_1_lap = dwt_forward_3d_lap(voxel_1_cuda)
        low_cuda_2_lap, highs_cuda_2_lap = dwt_forward_3d_lap(voxel_2_cuda)

        ### testing reconstruction
        voxels_reconstruction_lap_1 = dwt_inverse_3d_lap((low_cuda_1_lap, highs_cuda_1_lap))
        vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap_1.detach().cpu().numpy()[0, 0], 0.0)
        mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path_1) + '_orginal.off'))

        voxels_reconstruction_lap_2 = dwt_inverse_3d_lap((low_cuda_2_lap, highs_cuda_2_lap))
        vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap_2.detach().cpu().numpy()[0, 0], 0.0)
        mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path_2) + '_orginal.off'))

        for i in range(max_depth):

            # path 1
            highs_cuda_1_lap_new = highs_cuda_2_lap[:i+1] + highs_cuda_1_lap[i+1:]
            for level in skip_level:
                highs_cuda_1_lap_new[level] = torch.zeros(highs_cuda_1_lap_new[level].size()).to(device)
            voxels_reconstruction_lap_1 = dwt_inverse_3d_lap((low_cuda_1_lap, highs_cuda_1_lap_new))
            vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap_1.detach().cpu().numpy()[0, 0], 0.0)
            mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path_1) + f'_swap_{i}.off'))

            # path 2
            highs_cuda_2_lap_new = highs_cuda_1_lap[:i+1] + highs_cuda_2_lap[i+1:]
            for level in skip_level:
                highs_cuda_2_lap_new[level] = torch.zeros(highs_cuda_2_lap_new[level].size()).to(device)

            voxels_reconstruction_lap_2 = dwt_inverse_3d_lap((low_cuda_2_lap, highs_cuda_2_lap_new))
            vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap_2.detach().cpu().numpy()[0, 0], 0.0)
            mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path_2) + f'_swap_{i}.off'))