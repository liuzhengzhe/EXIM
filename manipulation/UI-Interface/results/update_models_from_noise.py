#!/usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:liruihui
@file: update_from_noise.py.py
@time: 2021/01/26
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description:
"""
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
    ## CONFIG
    selctions_file_path = []
    noises_path = r'D:\backup\Gallery_Lamp_1\selections'
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"
    cwd = os.getcwd()
    mlab = Matlab()
    mlab.start()
    mlab.run_code("parpool(4)")

    ## B
    BALL_SIZE = 0.02

    ## load models
    point_gen = PointGenerator()

    ## find the file names
    noises_names = list(glob.glob(os.path.join(noises_path, '*.npy')))
    noises_names = [ noises_name for noises_name in noises_names if os.path.basename(noises_name)[:6] == 'noises']

    ## create extra folder
    extra_folder = os.path.join(noises_path, 'updated_models')
    if not os.path.isdir(extra_folder):
        os.mkdir(extra_folder)

    for i in range(len(noises_names)):
        base_name_i = f"points_{os.path.basename(noises_names[i])[:-4].split('_')[1]}.xyz"
        file_name_i = os.path.join(noises_path, base_name_i)
        ## noise
        noise_1 = np.load(noises_names[i])
        points = point_gen.generate_points_from_noise(noise_1)[0]
        points = pc_normalize(points)

        ## load back the file
        points_i = np.loadtxt(file_name_i)
        points_i[:, :3] = points
        np.savetxt(os.path.join(extra_folder, base_name_i), points)

    print(f"Start {extra_folder}")
    res = mlab.run_func(script_path, extra_folder, BALL_SIZE)
    print(res)

    os.chdir(extra_folder)
    os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")