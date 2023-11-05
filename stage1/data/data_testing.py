from data.sampler import get_voxels, save_voxels
from scipy.io import loadmat
from utils.debugger import MyDebugger
import os
import numpy as np
from mesh_to_sdf import mesh_to_voxels

if __name__ == '__main__':

    debugger = MyDebugger('Voxels-Testing', is_save_print_to_file = False)
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
    for path in paths:
        mat = loadmat(path)
        voxels = get_voxels(mat, save_points= True, save_path = debugger.file_path(f'{os.path.basename(path)[:-4]}_original.xyz'))
        fft_signals = np.fft.fftn(voxels)

        for i in range(resolution // 2):
            fft_signals[resolution // 2 - i:resolution // 2 + i, :, :] = 0
            fft_signals[:, resolution // 2 - i:resolution // 2 + i, :] = 0
            fft_signals[:, :, resolution // 2 - i:resolution // 2 + i] = 0

            ifft_signals = np.real(np.fft.ifftn(fft_signals))

            save_voxels(debugger.file_path(f'{os.path.basename(path)[:-4]}_{i}_reconstructed.xyz'), ifft_signals)
            print(f"resolution {i} done!")
        print(f'done {path}!')


        print(voxels.shape)