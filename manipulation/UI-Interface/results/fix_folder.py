import glob
import os
import numpy as np
from pymatbridge import Matlab
if __name__ == '__main__':

    folder_path = r'C:\Users\student\Desktop\Interaction_UI\debug\Chair_2_part_composition_1\Figure_3\selection_color'
    mlab = Matlab()
    mlab.start()
    mlab.run_code("parpool(4)")
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"
    BALL_SIZE = 0.02

    res = mlab.run_func(script_path, folder_path, BALL_SIZE)
    print(res)

    os.chdir(folder_path)
    os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
