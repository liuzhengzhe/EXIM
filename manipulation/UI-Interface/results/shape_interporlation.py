from utils.debugger import MyDebugger
from PointSelector.PointGenerator import PointGenerator
from Model.model import pc_normalize
from Model.point_operation import plot_pcd_multi_rows
from pymatbridge import Matlab
from utils.utils import chamfer_distance
import os
import glob
import random
import numpy as np
import itertools


if __name__ == '__main__':
    debugger = MyDebugger("Interpolation")
    # mlab = Matlab()
    # mlab.start()
    # mlab.run_code("parpool(4)")

    ## CONFIG
    selctions_file_path = []
    noises_path = r'E:\backup_pointgen\Human_pose\selections'
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"
    cwd = os.getcwd()
    ## load models
    point_gen = PointGenerator()
    selections = [np.ones(point_gen.model.opts.np, np.int32)]
    # selections = [np.load(os.path.join(noises_path, 'selection_0.npy'))]
    # selections = [np.load(os.path.join(noises_path, f'selection_{i}.npy')) for i in range(1)]
    full_flag = True
    reverse_flag = False


    ## COLOR
    ## full color
    selections_color_1 = np.array([230, 185, 53]) / 255
    selections_color_2 = np.array([230, 68, 53]) / 255

    ## set 1
    # selections_color_1 = np.array([118, 121, 153]) / 255
    # selections_color_2 = np.array([186, 224, 221]) / 255
    ## set 1-2
    # selections_color_1 = np.array([29, 80, 185]) / 255 # BLUE
    # selections_color_2 = np.array([190, 32, 191]) / 255 # PURPLE
    ## set 1-3
    # selections_color_1 = np.array([68, 108, 204]) / 255 # BLUE
    # selections_color_2 = np.array([203, 67, 203]) / 255 # PURPLE
    ## set 1-4

    ## set 2
    # selections_color_2 = np.array([153, 118, 118]) / 255
    # selections_color_1 = np.array([224, 166, 166]) / 255
    # unselected_color = np.array([194, 235, 152]) / 255 # chair green first

    # Figure 2_4 color
    # unselected_color = np.array([252, 236, 126]) / 255 # orginal yellow
    # selections_color_1 = np.array([116, 148, 221]) / 255 # BLUE
    # selections_color_2 = np.array([217, 108, 217]) / 255 # PURPLE

    # Figure 2_5 color
    # unselected_color = np.array([254, 214, 126]) / 255 # orginal
    # selections_color_1 = np.array([130, 106, 210]) / 255 # BLUE
    # selections_color_2 = np.array([110, 186, 220]) / 255 # PURPLE

    ## Figure 2_6 color
    # unselected_color = np.array([194, 228, 115]) / 255 # orginal
    # selections_color_1 = np.array([252, 122, 131]) / 255 # BLUE
    # selections_color_2 = np.array([169, 103, 209]) / 255 # PURPLE

    ## Figure 2_7 color
    # unselected_color = np.array([117, 216, 107]) / 255 # orginal
    # selections_color_1 = np.array([254, 188, 129]) / 255 # BLUE
    # selections_color_2 = np.array([240, 120, 158]) / 255 # PURPLE

    ## Teaser 1 color
    # unselected_color = np.array([243, 173, 68]) / 255  # orginal
    # selections_color_1 = np.array([114, 183, 252]) / 255
    # selections_color_2 = np.array([41, 110, 180]) / 255 # PURPLE

    ## Teaser 2 color
    unselected_color = np.array([243, 173, 68]) / 255  # orginal
    # selections_color_1 = np.array([252, 114, 114]) / 255
    # selections_color_2 = np.array([181, 42, 42]) / 255 # PURPLE
    ## Thershold
    thershold = 0.0

    ## alpha range
    start_point = 0.0
    end_point = 1.0
    SAMPLE = 101
    use_latent = True
    special_samples = None
    # special_samples = [(0, 0), (1 / 3, 0.2), (2 / 3, 0.6), (1, 0.8)]
    # special_samples = [(0, 0), (1 / 5, 15 /100), (2 / 5, 40 / 100), (3 / 5, 60 / 100), (4 / 5, 75 / 100), (5 / 5, 100 / 100)]
    # special_samples = [(0, 0), (1 / 5, 8 /100), (2 / 5, 40 / 100), (3 / 5, 53 / 100), (4 / 5, 65 / 100), (5 / 5, 100 / 100)]
    # special_samples = [(0, 0), (1 / 5, 15 / 100), (2 / 5, 30 / 100), (3 / 5, 50 / 100), (4 / 5, 80 / 100),
    #                    (5 / 5, 100 / 100)]



    noises_names = list(glob.glob(os.path.join(noises_path, '*.npy')))
    noises_names = [ noises_name for noises_name in noises_names if os.path.basename(noises_name)[:6] == 'noises']
    if reverse_flag:
        noises_names = noises_names[::-1]
    files_names = list(glob.glob(os.path.join(noises_path, '*.xyz')))

    distances = [[0 for _ in range(len(noises_names))] for _ in range(len(noises_names))]
    ###
    for i in range(len(noises_names)):
        for j in range(i + 1, len(noises_names)):
            file_name_i = os.path.join(noises_path, f"points_{os.path.basename(noises_names[i])[:-4].split('_')[1]}.xyz")
            file_name_j = os.path.join(noises_path, f"points_{os.path.basename(noises_names[j])[:-4].split('_')[1]}.xyz")
            points_i = np.loadtxt(file_name_i)[:, :3]
            points_j = np.loadtxt(file_name_j)[:, :3]
            distance = chamfer_distance(points_i, points_j)
            distances[i][j] = distance

        print(f"{i} data is done")

    print("How many will be printed :", np.sum([[ distance > thershold for distance in disntance_item]for disntance_item in distances]))

    # setting the code interpolation
    indices = [[(i, j) for j in range(0 if not full_flag else i + 1, len(noises_names))] for i in range(len(noises_names))]
    indices = [item for sublist in indices for item in sublist]
    random.shuffle(indices)
    for i, j in indices:
        if i == j:
            continue

        ## logging
        if distances[i][j] < thershold or distances[j][i] < thershold:
            print(f"Skipped for {os.path.basename(noises_names[i])} {os.path.basename(noises_names[j])}....")
            continue
        if os.path.basename(noises_names[i])[:6] != 'noises' or os.path.basename(noises_names[j])[:6] != 'noises':
            continue
        print(f"Runing for {os.path.basename(noises_names[i])} {os.path.basename(noises_names[j])}....")
        ## noise
        noise_1 = np.load(noises_names[i])
        noise_2 = np.load(noises_names[j])

        ## create folder
        output_name = os.path.basename(noises_names[i])[:-4] + '_' + os.path.basename(noises_names[j])[:-4]
        folder_name = debugger.file_path(output_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        rows = len(selections)
        if special_samples is None:
            columns = SAMPLE
        else:
            columns = len(special_samples)

        pcds = [[None for _ in range(columns)] for _ in range(rows)]
        titles = [[None for _ in range(columns)] for _ in range(rows)]

        for idx, selection in enumerate(selections):
            if special_samples is None:
                samples = np.linspace(start_point, end_point, SAMPLE)
                samples = [((alpha - start_point) / (end_point - start_point), alpha) for alpha in samples]
            else:
                samples = special_samples
            for idx_2, (color_weight, alpha) in enumerate(samples):
                points = point_gen.interpolation_noise(noise_1, noise_2, selection, alpha, use_latent = use_latent)[0]
                points = pc_normalize(points)
                selection_expanded = np.tile(np.expand_dims(np.copy(selection), axis = 1), (1, 3))
                colors = selection_expanded * (selections_color_1[np.newaxis, :] * (1 - color_weight) + selections_color_2[np.newaxis, :] * color_weight) + (1 - selection_expanded) * unselected_color[np.newaxis, :]
                points_output = np.concatenate((points, colors), axis=1)

                write_name = f'{output_name}_selection_{idx}_{int(alpha*100)}.xyz'
                np.savetxt(os.path.join(folder_name, write_name), points_output)

                pcds[idx][idx_2] = points_output
                titles[idx][idx_2] = write_name[:-4]

        figure_name = debugger.file_path(f"{output_name}.png")
        plot_pcd_multi_rows(figure_name, pcds, titles)

        # res = mlab.run_func(script_path, folder_name)
        # print(res)

        # os.chdir(folder_name)
        # os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
        # os.chdir(cwd)


