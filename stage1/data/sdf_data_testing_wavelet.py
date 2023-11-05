from data.sampler import get_voxels, save_voxels
from scipy.io import loadmat
from utils.debugger import MyDebugger
import os
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
import mcubes
import pywt
from configs import config

if __name__ == '__main__':

    debugger = MyDebugger('Mesh_sdf-Wavlet-Testing', is_save_print_to_file = False)
    folder = r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627'
    cnt = 100
    # paths = [path for path in os.listdir(folder) if os.path.isdir(os.path.join(folder, path))]
    # paths = np.random.choice(paths, cnt, replace = False)
    paths = [
        # r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627\2a98a638f675f46e7d44dc16af152638',
        # r'Y:\proj51_backup\edward\ShapeNetCore.v1\02828884\f98acd1dbe15f3c02056b4bd5d870b47',
        r'E:\3D_models\dragon_recon\dragon_vrip.obj'
    ]
    resolution = 256


    ## TESTING
    padding_mode = 'zero'
    wavelet_type = 'bior6.8'
    wavelet = pywt.Wavelet(wavelet_type)
    skip_level = []

    for path in paths:
        print(f'start {path}!')
        save_path = os.path.join(config.backup_path, os.path.basename(path) + f'_{resolution}.npy')
        if os.path.exists(save_path):
            voxels = np.load(save_path)
        else:
            if os.path.isdir(path):
                mesh = trimesh.load_mesh(os.path.join(path, 'model_flipped_manifold.obj'))
            else:
                mesh = trimesh.load_mesh(path)
            voxels = mesh_to_voxels(mesh, resolution)
            np.save(save_path, voxels)
        print(f"saved path : {save_path}")

        vertices, traingles = mcubes.marching_cubes(voxels, 0.0)
        mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + '_original.off'))
        print("done coversion")

        coeffs = pywt.wavedecn(voxels, wavelet, mode=padding_mode)

        for i in range(len(coeffs) - 1): ### mass the details coefficent one by one

            if i in skip_level:
                continue

            level_coefficent = coeffs[len(coeffs) - 1 - i]
            for key, item in level_coefficent.items():
                coeffs[len(coeffs) - 1 - i][key] = np.zeros(item.shape)

            recons = pywt.waverecn(coeffs, wavelet, mode = padding_mode)


            vertices, traingles = mcubes.marching_cubes(recons,  0.0)
            mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + f'_{i}_reconstructed.off'))
            print(f"resolution {i} done!")
        print(f'done {path}!')

        print(voxels.shape)