
import glob
import os
from shutil import copyfile
from pymatbridge import Matlab

if __name__ == '__main__':

    folder_path = r'E:\backup_pointgen\human_animations'
    selection_folder = 'selections'
    mlab = Matlab()
    mlab.start()
    mlab.run_code("parpool(4)")
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"
    PER_PICTURE = 15
    BALL_SIZE = 0.02

    selection_path = os.path.join(folder_path, selection_folder)

    items = os.listdir(selection_path)
    cwd = os.getcwd()
    for item in items:
        folder_name = os.path.join(selection_path, item)
        if os.path.isdir(folder_name):
            print(f"Start {folder_name}")
            res = mlab.run_func(script_path, folder_name, BALL_SIZE)

            os.chdir(folder_name)
            os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
            os.chdir(cwd)