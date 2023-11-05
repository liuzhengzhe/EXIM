from data.sampler import get_voxels, save_voxels
from scipy.io import loadmat
from utils.debugger import MyDebugger
import os
import numpy as np
from mesh_to_sdf import mesh_to_voxels
import pywt
import copy

if __name__ == '__main__':

    debugger = MyDebugger('Voxels-Wavelets-Testing', is_save_print_to_file = False)
    postfix = '.off'
    folder = r'Y:\proj51_backup\edward\shapenet\modelBlockedVoxels256\03001627'
    cnt = 100
    # paths = [path for path in os.listdir(folder) if path.endswith(postfix)]
    # paths = np.random.choice(paths, cnt, replace = False)
    paths = [
        # r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627\2a98a638f675f46e7d44dc16af152638',
        r'Y:\proj51_backup\edward\shapenet\modelBlockedVoxels256\02828884\f98acd1dbe15f3c02056b4bd5d870b47.mat',
    ]
    resolution = 128



    ## TESTING
    padding_mode = 'zero'
    wavelet_type = 'db5'
    wavelet = pywt.Wavelet(wavelet_type)
    for path in paths:
        mat = loadmat(path)
        voxels = get_voxels(mat, save_points= True, save_path = debugger.file_path(f'{os.path.basename(path)[:-4]}_original.xyz'))

        coeffs = pywt.wavedecn(voxels, wavelet, mode = padding_mode)

        for i in range(len(coeffs) - 1): ### mass the details coefficent one by one
            level_coefficent = coeffs[len(coeffs) - 1 - i]
            for key, item in level_coefficent.items():
                coeffs[len(coeffs) - 1 - i][key] = np.zeros(item.shape)

            recons = pywt.waverecn(coeffs, wavelet, mode = padding_mode)
            save_voxels(debugger.file_path(f'{os.path.basename(path)[:-4]}_{i}_reconstructed.xyz'), recons)
            print(f"resolution {i} done!")

        print(f'done {path}!')


        print(voxels.shape)