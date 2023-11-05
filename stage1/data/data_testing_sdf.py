from data.sampler import get_voxels, save_voxels
from scipy.io import loadmat
from utils.debugger import MyDebugger
import os
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
import mcubes

def fftfreqs(res):
    freq = np.fft.fftfreq(res, d=1 / res)
    u, v, w = np.meshgrid(freq, freq, freq)
    u, v, w = u[:, :, :, np.newaxis], v[:, :, :, np.newaxis], w[:, :, :, np.newaxis]
    freqs = np.concatenate((u, v, w), axis=3)

    return freqs

def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res) # [dim0, dim1, dim2, d]
    dis = np.sqrt(np.sum(omega ** 2, axis=-1))
    filter_ = np.exp(-0.5*((sig*2*dis/res)**2))

    return filter_

if __name__ == '__main__':

    debugger = MyDebugger('Mesh_sdf-Testing', is_save_print_to_file = False)
    folder = r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627'
    cnt = 100
    # paths = [path for path in os.listdir(folder) if os.path.isdir(os.path.join(folder, path))]
    # paths = np.random.choice(paths, cnt, replace = False)
    paths = [
        r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627\2a98a638f675f46e7d44dc16af152638',
        # r'Y:\proj51_backup\edward\ShapeNetCore.v1\02828884\f98acd1dbe15f3c02056b4bd5d870b47',
    ]
    resolution = 256
    guassian_filter = True
    sig = 100


    freqs = fftfreqs(resolution)
    guassian = spec_gaussian_filter(resolution, sig)
    for path in paths:
        print(f'start {path}!')
        mesh = trimesh.load_mesh(os.path.join(path, 'model_flipped_manifold.obj'))
        voxels = mesh_to_voxels(mesh, resolution)
        vertices, traingles = mcubes.marching_cubes(voxels, 0.0)
        mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + '_original.off'))
        fft_signals = np.fft.fftn(voxels)
        if guassian_filter:
            fft_signals = fft_signals * guassian_filter

        for i in range(resolution // 2):
            fft_signals[resolution // 2 - i:resolution // 2 + i, :, :] = 0
            fft_signals[:, resolution // 2 - i:resolution // 2 + i, :] = 0
            fft_signals[:, :, resolution // 2 - i:resolution // 2 + i] = 0

            ifft_signals = np.real(np.fft.ifftn(fft_signals))

            vertices, traingles = mcubes.marching_cubes(ifft_signals,  0.0)
            mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + f'_{i}_reconstructed.off'))
            print(f"resolution {i} done!")
        print(f'done {path}!')

        print(voxels.shape)