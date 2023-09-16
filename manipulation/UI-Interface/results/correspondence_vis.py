
from utils.debugger import MyDebugger
from PointSelector.PointGenerator import PointGenerator
from Model.model import pc_normalize
from shutil import copyfile
import os
import numpy as np
from pymatbridge import Matlab
import math as m
import glob
from PIL import Image

def read_ball():
    ball_path = './template/2048.xyz'
    x = np.loadtxt(ball_path)
    ball = pc_normalize(x)

    N = ball.shape[0]
    # xx = torch.bmm(x, x.transpose(2,1))
    xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
    yy = xx.T
    xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
    dist = xy + xx + yy  # [B, N, N]

    # order = np.argsort(dist[1000])[::1]
    # ball = ball[order]

    return ball

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az

def compute_color_average(index_i, index_j, color_arr, cell = 2):

    results = np.zeros((index_i.shape[0], 3), dtype = np.float32)
    for i in range(-cell, cell+1):
        for j in range(-cell, cell+1):
            index_i_temp = ((index_i + i) + color_arr.shape[0]) % color_arr.shape[0]
            index_j_temp = ((index_j + j) + color_arr.shape[1]) % color_arr.shape[1]
            results += color_arr[index_i_temp, index_j_temp]

    results = results / ((2 * cell + 1) ** 2)
    return results

if __name__ == '__main__':


    ### config here
    folder_path = r'C:\Users\student\Desktop\Interaction_UI\debug\chair_corresespondence'
    selection_folder = 'selections'
    mlab = Matlab()
    mlab.start()
    mlab.run_code("parpool(4)")
    script_path = r"C:\Users\student\Desktop\Render\render\ball_withcolor.m"
    PER_PICTURE = 15

    # ## color ball
    ball  = read_ball()
    radius = np.max(np.sqrt(np.sum(ball ** 2, axis = 1)))
    #
    # theta = np.arccos(ball[:, 2] / radius) / np.pi
    # fine = (np.arctan2(ball[:, 1], ball[:, 0]) + np.pi) / (2 * np.pi)
    #
    # new_coordinate = np.concatenate((np.expand_dims(theta, axis = 1), np.expand_dims(fine, axis = 1)), axis = 1)
    #
    # ## compute coordinate
    # image = Image.open('./template/ziegler.png')
    # image = np.array(image)
    #
    # index_i = np.array(new_coordinate[:, 0] * image.shape[0], dtype= np.int32)
    # index_j = np.array(new_coordinate[:, 1] * image.shape[1], dtype= np.int32)
    # colors = compute_color_average(index_i, index_j, image, 3) / 255

    ## another colors
    # colors = (ball + 1) / 2 * 0.7 + 0.3
    colors = (ball + 1) / 2 * 0.4 + 0.6
    points_output = np.concatenate((ball, colors), axis=1)

    ## selection color
    selection_color = np.array([41, 110, 180]) / 255
    selection_color = np.tile(np.expand_dims(selection_color, axis = 0), (points_output.shape[0], 1))

    ##
    BALL_SIZE = 0.02
    BALL_SIZE_2 = 0.04




    selection_path = os.path.join(folder_path, selection_folder)
    changed_color_path = os.path.join(selection_path, 'correspondence_vis')
    if not os.path.isdir(changed_color_path):
        os.mkdir(changed_color_path)

    ball_path = os.path.join(changed_color_path, 'ball_folder')
    if not os.path.isdir(ball_path):
        os.mkdir(ball_path)

    # np.savetxt(os.path.join(ball_path, f'colored_ball_base.xyz'), points_output)


    for name in glob.glob(os.path.join(selection_path, '*.npy')):
        index = int(os.path.basename(name[:-4]).split("_")[1])
        subfolder_name = f'models_{index // PER_PICTURE}'
        subfolder_path = os.path.join(folder_path, subfolder_name)
        copyfile( os.path.join(subfolder_path, f'points_{index}.xyz'),
                  os.path.join(selection_path, f'points_{index}.xyz'))

        ## load back the xyz and write again
        points = np.loadtxt(os.path.join(selection_path, f'points_{index}.xyz'))
        colors_copyed = np.copy(colors)

        ## selection update
        selections = points[:, 3]
        colors_copyed[selections == 1, :] = selection_color[selections == 1, :]
        points = np.concatenate((points[:, :3], colors_copyed), axis = 1)

        np.savetxt(os.path.join(changed_color_path, f'points_{index}.xyz'), points)

        ## save the corresponding ball
        the_ball = np.concatenate((ball, colors_copyed), axis=1)
        np.savetxt(os.path.join(ball_path, f'colored_ball_{index}.xyz'), the_ball)

    ## run for the path
    res = mlab.run_func(script_path, changed_color_path, BALL_SIZE)
    res = mlab.run_func(script_path, ball_path, BALL_SIZE_2)
    print(res)

    ## PROCESS TO keySHOT
    # res = mlab.run_func(script_path, changed_color_path)
    os.chdir(changed_color_path)
    os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
    os.chdir(ball_path)
    os.system("for %a in ( \"*.off\" ) do meshlabserver -i \"%a\" -o \"%a.obj\" -m vc")
