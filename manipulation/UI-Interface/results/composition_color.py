import os
import numpy as np
import glob

if __name__ == '__main__':

    base_path = r'C:\Users\student\Desktop\Interaction_UI\debug\Chair_2_part_composition_1\Figure_3'
    unselected_color = np.array([243, 173, 68]) # orginal
    allow_unselected_flag = True

    pointcloud_file_pathes = glob.glob(os.path.join(base_path, '*.xyz'))
    for pointcloud_file_path in pointcloud_file_pathes:
        pointcloud_file_path = os.path.basename(pointcloud_file_path)
        ## the list of selection
        # selection_list = ['legs_3_selection.npy']
        # selection_list = ['seat_selection_3.npy']
        # selection_list = ['top_selection.npy', 'seat_selection_3.npy']
        # selection_list = ['top_selection.npy', 'seat_selection_3.npy', 'legs_3_selection.npy']
        # selection_list = [f'selection_{i}.npy' for i in  range(4)]
        # selection_list = ['buttom_selection.npy']
        # selection_list = ['top_selection.npy']
        selection_list = ['top_selection.npy', 'buttom_selection.npy']
        loaded_selections = [np.load(os.path.join(base_path, selection_file)) for selection_file in selection_list]

        loaded_pointcloud = np.loadtxt(os.path.join(base_path, pointcloud_file_path))

        final_selection_list = {i : -1 for i in range(loaded_pointcloud.shape[0])}

        # thershold
        thershold = 20
        for i in range(loaded_pointcloud.shape[0]):

            results = []
            for idx, loaded_selection in enumerate(loaded_selections):
                if loaded_selection[i] == 1:
                    results.append(idx + 1)

            if len(results) == 1:
                final_selection_list[i] = results[0]
            elif len(results) == 0 and allow_unselected_flag:
                final_selection_list[i] = 0



        ## porcess unknown one
        for i in range(loaded_pointcloud.shape[0]):
            if final_selection_list[i] == -1:
                distances = np.sum(np.abs(loaded_pointcloud[:, :3] - loaded_pointcloud[i, :3]), axis = 1)
                indices = np.argsort(distances)[:thershold]

                cnt_for_voting = [0 for _ in selection_list]
                for idx in indices:
                    if final_selection_list[idx] != -1:
                        cnt_for_voting[final_selection_list[idx] - 1] += 1

                result_index = np.argmax(cnt_for_voting) + 1
                final_selection_list[i] = result_index

        ## color 1
        # color_list = [unselected_color, np.array([152, 194, 235])] # blue only
        # color_list = [unselected_color, np.array([152, 235, 152])] # green only
        # color_list = [unselected_color, np.array([235, 152, 152])]  # red only
        color_list = [unselected_color, np.array([235, 152, 152]), np.array([152, 235, 152]), np.array([152, 194, 235]), np.array([235, 235, 152])]
        color_map = np.array([ color_list[final_selection_list[i]] for i in range(loaded_pointcloud.shape[0])])
        save_folder =os.path.join(base_path, 'selection_color')
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        output_points = np.concatenate((loaded_pointcloud[:, :3], color_map), axis = 1)
        np.savetxt(os.path.join(save_folder, pointcloud_file_path), output_points)