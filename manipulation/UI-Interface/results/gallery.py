from utils.debugger import MyDebugger
from PointSelector.PointGenerator import PointGenerator
from Model.model import pc_normalize
from Model.point_operation import plot_pcd_multi_rows
import os
import numpy as np

if __name__ == '__main__':

    debugger = MyDebugger("Gallery_Animals")

    ### config here
    SAMPLE_CNT = 120
    CLOUD_PER_ROW = 3
    CLOUD_ROW = 5

    ## load model here
    point_gen = PointGenerator()
    number_per_figure = CLOUD_PER_ROW * CLOUD_ROW

    ## selection
    selection = np.ones(point_gen.model.opts.np, np.int32)
    color_of_ball = np.array([195, 235, 152]) / 255

    for i in range((SAMPLE_CNT - 1) // number_per_figure + 1):

        ## make folder
        folder_path = debugger.file_path(f'models_{str(i)}')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        pcds = [[None for _ in range(CLOUD_PER_ROW)] for _ in range(CLOUD_ROW)]
        titles = [[None for _ in range(CLOUD_PER_ROW)] for _ in range(CLOUD_ROW)]

        for j in range(i * number_per_figure, min((i+1) * number_per_figure, SAMPLE_CNT)):
            points_arrs, point_noises = point_gen.generate_pointclouds(1)
            points = pc_normalize(points_arrs[0])
            colors = np.tile(np.expand_dims(np.copy(selection), axis=1), (1, 3)) * color_of_ball[np.newaxis, :]
            points_output = np.concatenate((points, colors), axis=1)
            np.savetxt(os.path.join(folder_path, f'points_{j}.xyz'), points_output)
            np.save(os.path.join(folder_path, f'noises_{j}.npy'), point_noises[0])

            ## set pcd
            index = j - i * number_per_figure
            pcds[index // CLOUD_PER_ROW][index % CLOUD_PER_ROW] = points
            titles[index // CLOUD_PER_ROW][index % CLOUD_PER_ROW] = f"points_{j}"


        figure_name = debugger.file_path(f"set_{i}.png")
        plot_pcd_multi_rows(figure_name, pcds, titles)



    # points_arrs, points_noise = point_gen.generate_pointclouds(100)