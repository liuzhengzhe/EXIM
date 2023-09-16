import glob
import os
import numpy as np
from pymatbridge import Matlab
if __name__ == '__main__':

    folder_path = r'C:\Users\student\Desktop\Interaction_UI\debug\2021-01-22_20-15-42_Main_Interface\10_11'
    mlab = Matlab()
    mlab.start()
    mlab.run_code("parpool(4)")
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"

    colors_to_replace = np.array([235, 194, 152]) / 255

    file_names = glob.glob(os.path.join(folder_path, '*.xyz'))
    for file_name in file_names:
        points = np.loadtxt(file_name)
        expanded_colors = np.tile(np.expand_dims(colors_to_replace, axis = 0), (points.shape[0], 1))
        points = np.concatenate((points[:, :3], expanded_colors), axis = 1)
        np.savetxt(file_name, points)

    res = mlab.run_func(script_path, folder_path)
    print(res)

    os.chdir(folder_path)
    os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
