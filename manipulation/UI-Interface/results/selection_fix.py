
import glob
import os
import numpy as np
from shutil import copyfile
from pymatbridge import Matlab

if __name__ == '__main__':

    folder_path = r'C:\Users\student\Desktop\Interaction_UI\debug\figure_2_animals'
    selection_folder = 'selections'
    mlab = Matlab()
    mlab.start()
    mlab.run_code("parpool(4)")
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"
    PER_PICTURE = 15
    BALL_SIZE = 0.02
    #
    # colors = np.array([194, 235, 152]) / 255 # green chair
    # colors = np.array([235, 152, 194]) / 255 # pink guitar
    # colors = np.array([235, 152, 152]) / 255 # red lamp
    # colors = np.array([152, 235, 235]) / 255 # Celeste vessels
    # colors = np.array([152, 152, 235]) # Blue purple Cars --> scale up the ball a bit
    # colors = np.array([235, 194, 152]) / 255 # Orange Airplane
    # colors = np.array([152, 194, 235]) / 255 # pink blue bottle
    colors = np.array([243, 173, 68]) / 255  # orginal
    ## HSB : 30 35 92
    ## selection color set 1
    # unselected_color = np.array([254, 214, 126]) / 255  # orginal
    # selections_color = np.array([110, 186, 220]) / 255 # PURPLE

    ## selection color set 2
    unselected_color = np.array([243, 173, 68]) / 255  # orginal
    selections_color = np.array([41, 110, 180]) / 255 # teaser 1
    selection_flag = False
    selection_list = None

    selection_path = os.path.join(folder_path, selection_folder)
    noises_names = list(glob.glob(os.path.join(selection_path, '*.npy')))
    noises_names = [ noises_name for noises_name in noises_names if os.path.basename(noises_name)[:6] == 'noises']
    for name in noises_names:
        index = int(os.path.basename(name[:-4]).split("_")[1])
        subfolder_name = f'models_{index // PER_PICTURE}'
        subfolder_path = os.path.join(folder_path, subfolder_name)
        copyfile( os.path.join(subfolder_path, f'points_{index}.xyz'),
                  os.path.join(selection_path, f'points_{index}.xyz'))

        ## load back the xyz and write again
        points = np.loadtxt(os.path.join(selection_path, f'points_{index}.xyz'))
        if not selection_flag:
            colors_expanded = np.tile(np.expand_dims(colors, axis = 0), (points.shape[0], 1))
        else:
            unselected_color_expanded = np.tile(np.expand_dims(unselected_color, axis = 0), (points.shape[0], 1))
            selected_color_expanded = np.tile(np.expand_dims(selections_color, axis = 0), (points.shape[0], 1))
            selection = points[:, 3]
            unselected_color_expanded[selection == 1, :] = selected_color_expanded[selection == 1, :]
            colors_expanded = unselected_color_expanded

        points = np.concatenate((points[:, :3], colors_expanded), axis = 1)
        np.savetxt(os.path.join(selection_path, f'points_{index}.xyz'), points)
    #
    res = mlab.run_func(script_path, selection_path, BALL_SIZE)
    print(res)

    os.chdir(selection_path)
    os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
