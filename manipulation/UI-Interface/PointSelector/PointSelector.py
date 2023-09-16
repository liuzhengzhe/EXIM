import numpy as np
import glm
import math
import random
import torch
from OpenGL.GL import *
from OpenGL.GLUT import *
from utils.utils import load_off, compute_vertices_normal
from part_seg.pointnet2_part_seg_msg import get_model, seg_classes, seg_label_to_cat, index_to_class, to_categorical
from utils.utils import check_point_inside_polygon
SHIFT_KEY = 16777248
OPT_KEY = 16777251
MOUSE_LEFT_BUTTON = 1
MOUSE_RIGHT_BUTTON = 2

EPS = 1e-7
class PointSelector(object):

    def __init__(self, winW : int, winH : int, use_part_seg : bool = True, points = None, initial_Fov = 45.0, ball_size = 0.015):
        self.pointcloud_arr = None
        self.pointcloud_selected = None

        ## init pointcloud
        if points is None:
            self.load_PointCloud('./data/1.xyz')
        else:
            self.pointcloud_arr = np.array(points, dtype=np.float32)

        assert self.pointcloud_arr is not None
        self.pointcloud_selected = np.zeros(self.pointcloud_arr.shape[0], dtype=np.int32)

        ## log
        print("Point cloud shapes:")
        print(self.pointcloud_arr.shape)
        print(self.pointcloud_selected.shape)
        print(self.pointcloud_arr.strides)

        ## polygons
        self.selection_polygon = []

        ## collection of motion records
        self.prevMouseX = 0
        self.prevMouseY = 0
        self.pressed_mouse_button = 0
        self.pressed_keyboard_button = 0
        self.windowW = winW
        self.windowH = winH
        self._rotate_scale = 0.5
        self.currFovy = initial_Fov
        self._near = 0.1
        self._far = 200
        self.polygon_delta = 0.02

        ## definite initial view matrix and projection matrix
        mean_position = -np.mean(self.pointcloud_arr, axis=0)
        model_matrix = glm.translate(glm.mat4(), glm.vec3(mean_position[0], mean_position[1], mean_position[2]))
        # self.modelview_matrix = model_matrix
        self.eye_position = glm.vec3(-1.8, 1.8, -1.8)
        self.modelview_matrix = glm.lookAt(self.eye_position, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))  # * model_matrix
        # self.projection_matrix = glm.mat4()
        self.update_provy(0)
        self.reset_projection()

        ## part seg
        self.use_part_seg = use_part_seg
        self.part_seg_model = None
        self.points_segmentation = np.full(self.pointcloud_selected.shape, -1, dtype = np.int32)
        self.label = 4

        ## selection mode
        self.selection_mode = 1

        ## class and parts
        model_path = './part_seg/models/best_model_chairs_new.pth'
        self.num_part = 50
        self.num_class = 16
        if self.use_part_seg:
            print("Start Loading Segmentation!!!")
            self.part_seg_model = get_model(self.num_part, normal_channel = False).cuda()
            checkpoint = torch.load(model_path)
            self.part_seg_model.load_state_dict(checkpoint['model_state_dict'])
            print("Finish Loading segmentation !!!!!")
            self.points_segmentation = self.predict_segmentation()


        ## ball
        self.current_ball_size = ball_size
        ball_positions, _, ball_indices = load_off('./template/ball_uniform.off')
        ball_positions = ball_positions
        ball_positions, ball_indices = np.array(ball_positions, dtype=np.float32), np.array(ball_indices,
                                                                                            dtype=np.int32)
        ball_normals = np.array(compute_vertices_normal(ball_positions, ball_indices), dtype=np.float32)
        self.ball_template_base = (ball_positions, ball_indices, ball_normals)
        self.ball_template = None
        self.update_ball()

    def update_ball(self):
        ball_positions = self.ball_template_base[0] * self.current_ball_size
        ball_normals = np.array(compute_vertices_normal(ball_positions, self.ball_template_base[1]), dtype=np.float32)
        self.ball_template = (ball_positions, self.ball_template_base[1], ball_normals)


    def predict_segmentation(self):
        assert self.pointcloud_arr is not None
        print("Predicting Segmentation!!")

        ## vote cnt
        vote_cnt = 5
        
        ## predict
        points_arr_torch = torch.from_numpy(self.pointcloud_arr).float().cuda()
        points_arr_torch = points_arr_torch.transpose(1, 0)
        label_torch = torch.from_numpy(np.array(self.label)).long().cuda()

        ## get prediction
        assert self.part_seg_model is not None

        final_logits = np.zeros((self.pointcloud_arr.shape[0], self.num_part))
        for i in range(vote_cnt):
            preidct_seg, _ = self.part_seg_model(points_arr_torch.unsqueeze(0), to_categorical(label_torch.unsqueeze(0), self.num_class))
            logits = preidct_seg[0].detach().cpu().numpy()
            final_logits += logits

        final_logits = final_logits / vote_cnt

        ## get corresponding part prediction
        seg_class = index_to_class[self.label]
        range_pred = seg_classes[seg_class]
        final_logits = final_logits[:, range_pred]
        predict = np.argmax(final_logits, axis = 1)

        print("Done Predicting Segmentation!!")
        return np.array(predict, dtype = np.int32)

    def load_PointCloud(self, path : str):
        self.pointcloud_arr = np.loadtxt(path, skiprows= 0, max_rows= 100000, dtype= np.float32)

    def update_pointcloud(self, points):
        self.pointcloud_arr = np.array(points, dtype = np.float32)
        if self.use_part_seg:
            self.points_segmentation = self.predict_segmentation()

    def reset_selection(self):
        self.pointcloud_selected = np.zeros(self.pointcloud_selected.shape, dtype = np.int32)

    def assign_selection(self, selection):
        self.pointcloud_selected = np.array(selection, dtype = np.int32)

    def motion_function(self, x, y):
        y = self.windowH - 1 - y

        dx = x - self.prevMouseX
        dy = y - self.prevMouseY

        print(dx, dy)

        if (dx == 0 and dy == 0):
            return

        ## update location
        self.prevMouseX = x
        self.prevMouseY = y

        print(f"pressed mouse {self.pressed_mouse_button == MOUSE_LEFT_BUTTON}")
        print(f"pressed keyboard {self.pressed_keyboard_button == OPT_KEY}")
        if self.pressed_mouse_button == MOUSE_LEFT_BUTTON and self.pressed_keyboard_button == SHIFT_KEY:
            tx = 0.01 * dx * self.currFovy / 90.0
            ty = 0.01 * dy * self.currFovy / 90.0

            self.modelview_matrix = glm.translate(glm.mat4(), glm.vec3(tx, ty, 0.0)) * self.modelview_matrix
            print("New modelview matrx :", self.modelview_matrix)


            return True
        elif self.pressed_mouse_button == MOUSE_LEFT_BUTTON and self.pressed_keyboard_button == 0:
            nx = -dy
            ny = dx
            scale = math.sqrt(nx * nx + ny * ny)

            nx = nx / scale
            ny = ny / scale
            angle = scale * self._rotate_scale * self.currFovy / 90.0

            ## the matrix
            print(self.modelview_matrix)
            to_translate = glm.vec3(self.modelview_matrix[3, 0], self.modelview_matrix[3, 1], self.modelview_matrix[3, 2])
            inverse_translate_matrix = glm.translate(glm.mat4(), -to_translate)
            rotate_matrix = glm.rotate(glm.mat4(), glm.radians(angle), glm.vec3(nx, ny, 0.0))
            translate_matrix = glm.translate(glm.mat4(), to_translate)
            self.modelview_matrix = translate_matrix * rotate_matrix * inverse_translate_matrix * self.modelview_matrix
            print("ModelView matrix old method :", self.modelview_matrix)



            return True
        elif self.pressed_mouse_button == MOUSE_LEFT_BUTTON and self.pressed_keyboard_button == OPT_KEY:
            self.update_provy(dy / self.windowH)
            print(self.currFovy)
            self.reset_projection()
            return True
        elif self.pressed_mouse_button == MOUSE_RIGHT_BUTTON:
            self.add_selection_polygon(x, y)
            return True

        return False

    def reset_projection(self):
        print("current fovy", self.currFovy)
        self.projection_matrix = glm.perspective(glm.radians(self.currFovy), self.windowW / self.windowH, self._near, self._far)

    def set_prev_mouse_pos(self, x, y):
        self.prevMouseX = x
        self.prevMouseY = self.windowH - 1 - y

    def add_selection_polygon(self, x, y):
        ## compute the corresponding coordinate
        x_ratio = x / self.windowW
        y_ratio = y / self.windowH

        ## compute_
        x_NDC = x_ratio * 2 - 1
        y_NDC = y_ratio * 2 - 1

        ##

        if len(self.selection_polygon) == 0 or \
                abs(x_NDC - self.selection_polygon[-1][0]) +\
                abs(y_NDC - self.selection_polygon[-1][1]) > self.polygon_delta: # 1-norm
            self.selection_polygon.append((x_NDC, y_NDC))

    def update_provy(self, dx):

        _FOVY_K = 0.005

        if self.currFovy < _FOVY_K:
            x = math.log10(self.currFovy) + _FOVY_K - math.log(_FOVY_K)
        else:
            x = self.currFovy

        # add in the x-space
        x += dx * 10

        ##
        if x > 0:
            if x > 179.9:
                x = 179.9
        else:
            x = math.pow(10, x - _FOVY_K + math.log(_FOVY_K))
            x = max(1e-7, x)

        self.currFovy = x

    def update_selection(self):

        ## compute the NDC for all points
        point_arrs = np.concatenate((self.pointcloud_arr, np.ones((self.pointcloud_arr.shape[0], 1))), axis = 1)
        point_arrs_camera = np.matmul(point_arrs, np.array(self.modelview_matrix))
        point_arrs_NDC = np.matmul(point_arrs_camera, np.array(self.projection_matrix))
        point_arrs_NDC = point_arrs_NDC / (np.expand_dims(point_arrs_NDC[:, 3], axis = 1) + EPS)

        ## compute inside outside
        for i in range(self.pointcloud_arr.shape[0]):
            if check_point_inside_polygon(point_arrs_NDC[i, : 2], self.selection_polygon):
                self.pointcloud_selected[i] = self.selection_mode






if __name__ == '__main__':
    point_select = PointSelector()

