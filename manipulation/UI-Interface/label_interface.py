from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from UI.openglwidget import OpenGLWidget
from PointSelector.PointSelector import MOUSE_RIGHT_BUTTON
from PointSelector.PointGenerator import PointGenerator
from utils.debugger import MyDebugger
from Model.model import pc_normalize
import numpy as np
import os
import sys
import traceback
import trimesh

def scale_to_unit_cube(mesh, scale_ratio = 0.45):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)
    vertices *= scale_ratio
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

class MainInterface(QMainWindow):
    def __init__(self, debugger, use_part_seg, use_two_screen=True, use_point_gen=False):
        super().__init__()

        # set the title of main window
        self.debugger = debugger
        self.use_point_gen = use_point_gen
        self.use_two_screen = use_two_screen
        self.use_part_seg = use_part_seg
        self.setWindowTitle('Main Interface')

        # set the size of window
        self.Width = 1000
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)

        # add all widgets
        self.sidebar_main_subwidgets = []
        self.sidebar_control_subwidgets = []

        ## init for sidebar
        self.point_gen = None
        self.sidebar_widget = None
        self.main_opengl_widget = None
        self.another_opengl_widget = None
        self.main_widget = None

        ## if use gen
        if self.use_point_gen:
            self.point_gen = PointGenerator()

        ## define noise
        self.noises = None
        self.points = None
        self.alpha = 0.0
        self.MAX_ALPHA = 100
        self.STEP = 5
        self.pointLight_angle = 0

        ##
        self.pointcloud_indices = [-1] * 2 if self.use_two_screen else [-1]
        self.pointcloud_cnt = 0
        self.interpolation_cnt = 0
        self.current_noise = None
        self.my_edit_pc = None
        self.my_mesh = None

        ##
        self.INITIAL_FOV = 45.0
        self.MAX_BALL_SIZE = 50
        self.INITIAL_BALL_SIZE = 20
        self.BALL_RATIO = 1000

        ## current direction_vector
        self.current_noise_dir = None
        self.current_noise_dir_ratio = 0.0
        self.MAX_NOISE_RATIO = 2.0
        self.MAX_SCALE_NOISE = 100
        self.update_noise_search_dir()

        ## another index for storing the thing
        self.partial_noise_index = 0
        self.selection_cnt = 0

        self.testing_mode = False

        ## current folder name
        self.current_folder_name = 'noise_-1'

        self.initUI()

    def initUI(self):

        ## sidebar
        self.init_sidebar()

        ## opengl part
        if self.use_point_gen:
            sample_cnt = 2 if self.use_two_screen else 1
            points, noises = self.point_gen.generate_pointclouds(sample_cnt)
            self.main_opengl_widget = OpenGLWidget(use_part_seg=self.use_part_seg, points=points[0],
                                                   initial_fov=self.INITIAL_FOV,
                                                   ball_size=self.INITIAL_BALL_SIZE / self.BALL_RATIO)
            update_cnt = 1
            if self.use_two_screen:
                self.another_opengl_widget = OpenGLWidget(use_part_seg=False, points=points[1],
                                                          initial_fov=self.INITIAL_FOV,
                                                          ball_size=self.INITIAL_BALL_SIZE / self.BALL_RATIO)
                self.pointcloud_indices[1] = self.pointcloud_cnt + 1
                update_cnt += 1

            ## update reocrds
            self.points = points
            self.noises = noises
            self.pointcloud_indices[0] = self.pointcloud_cnt
            self.pointcloud_cnt += update_cnt
            self.save_basic_points_info()
        else:
            self.main_opengl_widget = OpenGLWidget(use_part_seg=self.use_part_seg, initial_fov=self.INITIAL_FOV,
                                                   ball_size=self.INITIAL_BALL_SIZE / self.BALL_RATIO)
            if self.use_two_screen:
                self.another_opengl_widget = OpenGLWidget(use_part_seg=False, initial_fov=self.INITIAL_FOV,
                                                          ball_size=self.INITIAL_BALL_SIZE / self.BALL_RATIO)
            self.pointcloud_indices[0] = self.pointcloud_cnt
            self.pointcloud_cnt += 1

        ## main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar_widget)
        main_layout.addWidget(self.main_opengl_widget)
        main_layout.setStretch(0, 20)
        main_layout.setStretch(1, 200)
        if self.another_opengl_widget is not None:
            main_layout.addWidget(self.another_opengl_widget)
            main_layout.setStretch(2, 200)

        # setting the main widget
        self.main_widget = QWidget()
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.main_widget)

        ## setting the lighting position
        self.update_light_position(45)

    ######## SIDEBAR CALLBACK ######################
    def init_sidebar_functions(self):

        ## Generate new pair #0
        button = QPushButton(f"Generate New {'Shape' if not self.use_two_screen else 'Pairs'}", self)
        button.clicked.connect(self.generate_new_noise)
        self.sidebar_main_subwidgets.append(button)
        button.setVisible(False)

        ## interface for generations
        if self.use_point_gen and self.use_two_screen:

            ## Generate all #1
            checkbox = QCheckBox("Generate Both Models", self)
            checkbox.setCheckState(Qt.Checked)
            self.sidebar_main_subwidgets.append(checkbox)

            ## slider for interpolation #3
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(0, self.MAX_ALPHA)
            slider.setValue(0)
            slider.setSingleStep(self.STEP)
            slider.setPageStep(self.STEP)
            slider.valueChanged.connect(self.update_alpha)

            ## alpha label #2
            label = QLabel(str(slider.value()), self)
            label.setAlignment(Qt.AlignCenter)

            self.sidebar_main_subwidgets.append(label)
            self.sidebar_main_subwidgets.append(slider)

            ## latent space #4
            checkbox = QCheckBox("Latent Space", self)
            checkbox.setCheckState(Qt.Checked)
            self.sidebar_main_subwidgets.append(checkbox)
            if self.testing_mode:
                checkbox.setVisible(False)

            ## Button for the interpolation #5
            self.direction_checkbox = QCheckBox("Left to Right")
            self.direction_checkbox.setCheckState(Qt.Checked)
            self.sidebar_main_subwidgets.append(self.direction_checkbox)

            ## Button for the interpolation #6
            button = QPushButton("Interpolate")
            button.clicked.connect(self.interpolate_pointclouds)
            self.sidebar_main_subwidgets.append(button)

            ## Button for update noise
            button = QPushButton("Update noise after interpolation")
            button.clicked.connect(self.update_noise)
            self.sidebar_main_subwidgets.append(button)
            if self.testing_mode:
                button.setVisible(False)

            ## slide bar for update part of the noise
            self.noise_slider = QSlider(Qt.Horizontal, self)
            self.noise_slider.setRange(-self.MAX_SCALE_NOISE, self.MAX_SCALE_NOISE)
            self.noise_slider.setValue(0)
            self.noise_slider.setSingleStep(self.STEP)
            self.noise_slider.setPageStep(self.STEP)
            self.noise_slider.valueChanged.connect(self.change_model_dir)
            self.sidebar_main_subwidgets.append(self.noise_slider)
            if self.testing_mode:
                self.noise_slider.setVisible(False)

            ## Button for update noise direction
            button = QPushButton("Update noise search dir")
            button.clicked.connect(self.update_noise_search_dir)
            if self.testing_mode:
                button.setVisible(False)
            else:
                self.sidebar_main_subwidgets.append(button)

            ## Button for update the position after searching
            button = QPushButton("Update noise by search dir")
            button.clicked.connect(self.update_noise_search_result)
            self.sidebar_main_subwidgets.append(button)
            if self.testing_mode:
                button.setVisible(False)
            else:
                self.sidebar_main_subwidgets.append(button)

            ## save the partial selection noise
            button = QPushButton("Save partial noise")
            button.clicked.connect(self.save_current_selection_noise)
            self.sidebar_main_subwidgets.append(button)
            if self.testing_mode:
                button.setVisible(False)
            else:
                self.sidebar_main_subwidgets.append(button)

        ## replace random noise
        button = QPushButton("Update random Noise")
        button.clicked.connect(self.random_update_noises)
        self.sidebar_main_subwidgets.append(button)
        button.setVisible(False)

        ## Normal functions
        button = QPushButton("Unselect All")
        button.clicked.connect(self.unselect_all)
        self.sidebar_main_subwidgets.append(button)

        ## Selection model
        self.selection_mode_checkbox = QCheckBox("Selection Mode")
        self.selection_mode_checkbox.stateChanged.connect(self.update_selection_mode)
        self.selection_mode_checkbox.setCheckState(Qt.Checked)
        self.sidebar_main_subwidgets.append(self.selection_mode_checkbox)

        ## Define checker for loading
        self.load_checkbox = QCheckBox("Load/Save Left Model", self)
        self.load_checkbox.setCheckState(Qt.Checked)
        if self.use_two_screen:
            self.sidebar_main_subwidgets.append(self.load_checkbox)
        else:
            self.load_checkbox.setVisible(False)

        ## Define save the noise and model
        button = QPushButton("Save the model")
        button.clicked.connect(self.save_for_gallery)
        self.sidebar_main_subwidgets.append(button)
        button.setVisible(False)

        button = QPushButton("Load point cloud to get mesh")
        button.clicked.connect(self.load_noise)
        self.sidebar_main_subwidgets.append(button)

        ## load selection
        button = QPushButton("Load selected Mesh")
        button.clicked.connect(self.load_selection)
        self.sidebar_main_subwidgets.append(button)

        ## save selection
        button = QPushButton("Save selected point cloud")
        button.clicked.connect(self.save_tail_selection)
        self.sidebar_main_subwidgets.append(button)

        ## save selection
        button = QPushButton("Save Selection (Head)")
        button.clicked.connect(self.save_head_selection)
        self.sidebar_main_subwidgets.append(button)

        ## save selection
        button = QPushButton("Save Selection (Feet)")
        button.clicked.connect(self.save_feet_selection)
        self.sidebar_main_subwidgets.append(button)

        ## save selection
        button = QPushButton("Save selected Mesh Part")
        button.clicked.connect(self.save_body_selection)
        self.sidebar_main_subwidgets.append(button)

        ## save selection
        button = QPushButton("Add Selection")
        button.clicked.connect(self.add_selection)
        self.sidebar_main_subwidgets.append(button)
        # if self.testing_mode:
        #     button.setVisible(False)
        # else:
        #     self.sidebar_main_subwidgets.append(button)

        ## load partial selection
        button = QPushButton("Load Partial Noise")
        button.clicked.connect(self.load_noise_on_selection)
        self.sidebar_main_subwidgets.append(button)
        button.setVisible(False)

        ## Toggle selection
        button = QPushButton("Toggle Selection")
        button.clicked.connect(self.toggle_selection)
        self.sidebar_main_subwidgets.append(button)

        ## Define Lighting
        label = QLabel("Light Location", self)
        label.setAlignment(Qt.AlignCenter)
        self.sidebar_control_subwidgets.append(label)

        ## Define slider
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, self.MAX_ALPHA)
        slider.setValue(50)
        slider.setSingleStep(self.STEP)
        slider.setPageStep(self.STEP)
        slider.valueChanged.connect(self.update_light_position)
        self.sidebar_control_subwidgets.append(slider)

        ## Define Lighting
        label = QLabel("FOV", self)
        label.setAlignment(Qt.AlignCenter)
        self.sidebar_control_subwidgets.append(label)

        ## slider
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, 180)
        slider.setValue(int(self.INITIAL_FOV))
        slider.setSingleStep(self.STEP)
        slider.setPageStep(self.STEP)
        slider.valueChanged.connect(self.update_FOV)
        self.sidebar_control_subwidgets.append(slider)

        ## update
        button = QPushButton("Update Selected Color")
        button.clicked.connect(self.update_selected_color)
        self.sidebar_control_subwidgets.append(button)

        ## update
        button = QPushButton("Update Unselected Color")
        button.clicked.connect(self.update_unselected_color)
        self.sidebar_control_subwidgets.append(button)

        ## Define Lighting
        label = QLabel("Ball Size", self)
        label.setAlignment(Qt.AlignCenter)
        self.sidebar_control_subwidgets.append(label)

        ## slider
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, 50)
        slider.setValue(int(self.INITIAL_BALL_SIZE))
        slider.setSingleStep(self.STEP)
        slider.setPageStep(self.STEP)
        slider.valueChanged.connect(self.updateBallSize)
        self.sidebar_control_subwidgets.append(slider)

        for widget in self.sidebar_main_subwidgets:
            if hasattr(widget, 'setFont'):
                widget.setFont(QFont('Times', 15))

        for widget in self.sidebar_control_subwidgets:
            if hasattr(widget, 'setFont'):
                widget.setFont(QFont('Times', 15))

    ######### INTERACTION ##########################
    def keyPressEvent(self, event: QKeyEvent):
        print('press key', event.key())
        MainInterface.keyPressCallback(self.main_opengl_widget, event)
        if self.another_opengl_widget is not None:
            MainInterface.keyPressCallback(self.another_opengl_widget, event)

    @staticmethod
    def keyPressCallback(widget: OpenGLWidget, event: QKeyEvent):
        widget.point_selector.pressed_keyboard_button = event.key()

    def keyReleaseEvent(self, event: QKeyEvent):
        print('release key', event.key())
        MainInterface.keyReleaseCallback(self.main_opengl_widget)
        if self.another_opengl_widget is not None:
            MainInterface.keyReleaseCallback(self.another_opengl_widget)

    @staticmethod
    def keyReleaseCallback(widget: OpenGLWidget):
        widget.point_selector.pressed_keyboard_button = 0

    def mousePressEvent(self, event: QMouseEvent):
        print('mouse press at', event.pos())
        print('key', event.button())

        ## main
        MainInterface.mousePressCallback(self.main_opengl_widget, event)
        if self.another_opengl_widget is not None:
            MainInterface.mousePressCallback(self.another_opengl_widget, event)

    @staticmethod
    def mousePressCallback(widget: OpenGLWidget, event: QMouseEvent):
        widget.point_selector.pressed_mouse_button = event.button()

        ## setting
        x = event.pos().x() - widget.geometry().topLeft().x()
        y = event.pos().y() - widget.geometry().topLeft().y()
        if x >= 0 and x <= widget.width() and y >= 0 and y <= widget.height():
            widget.point_selector.set_prev_mouse_pos(x, y)

    def mouseReleaseEvent(self, event: QMouseEvent):
        print('mouse release at', event.pos())
        print('key', event.button())

        self.mouseReleaseCallback(self.main_opengl_widget)
        if self.another_opengl_widget is not None:
            self.mouseReleaseCallback(self.another_opengl_widget)

    def mouseReleaseCallback(self, widget: OpenGLWidget):

        ## repaint once
        widget.repaint()
        if widget.point_selector.pressed_mouse_button == MOUSE_RIGHT_BUTTON:
            if len(widget.point_selector.selection_polygon) >= 3:
                widget.point_selector.update_selection()
                widget.update_selected_points()
                widget.point_selector.selection_polygon.clear()
                widget.repaint()

                ## update the selection of another one
                if self.main_opengl_widget == widget and self.use_two_screen:
                    MainInterface.copy_selection(widget, self.another_opengl_widget)
                if self.another_opengl_widget == widget and self.use_two_screen:
                    MainInterface.copy_selection(widget, self.main_opengl_widget)

        widget.point_selector.pressed_mouse_button = 0

    def mouseMoveEvent(self, event: QMouseEvent):
        print('mouse move at', event.pos())
        print('key', event.button())

        ## MAIN SCREEN
        MainInterface.mouseMoveCallback(self.main_opengl_widget, event)
        if self.another_opengl_widget is not None:
            MainInterface.mouseMoveCallback(self.another_opengl_widget, event)

    @staticmethod
    def mouseMoveCallback(widget: OpenGLWidget, event: QMouseEvent):
        x = event.pos().x() - widget.geometry().topLeft().x()
        y = event.pos().y() - widget.geometry().topLeft().y()
        if x >= 0 and x <= widget.width() and y >= 0 and y <= widget.height():
            repaint_flag = widget.point_selector.motion_function(x, y)
            if repaint_flag:
                widget.repaint()

    def generate_new_noise(self):
        print("Entered this function!")
        only_right = self.use_two_screen and not self.sidebar_main_subwidgets[1].isChecked()
        if only_right:
            sample_cnt = 1
            points, noises = self.point_gen.generate_pointclouds(sample_cnt)
            assert self.another_opengl_widget is not None
            MainInterface.update_pointcloud(self.another_opengl_widget, points[0])
            self.noises[1] = noises[0]
            self.points[1] = points[0]

            ## update index
            self.pointcloud_indices[1] = self.pointcloud_cnt
            self.pointcloud_cnt += 1
            self.interpolation_cnt = 0
            self.save_basic_points_info()

        else:
            sample_cnt = 2 if self.use_two_screen else 1
            points, noises = self.point_gen.generate_pointclouds(sample_cnt)
            MainInterface.update_pointcloud(self.main_opengl_widget, points[0])
            self.pointcloud_indices[0] = self.pointcloud_cnt
            update_cnt = 1
            if self.another_opengl_widget is not None:
                MainInterface.update_pointcloud(self.another_opengl_widget, points[1])
                self.pointcloud_indices[1] = self.pointcloud_cnt + 1
                update_cnt += 1
            self.noises = noises
            self.points = points
            self.interpolation_cnt = 0
            self.save_basic_points_info()

            ## update index
            self.pointcloud_cnt += update_cnt

    @staticmethod
    def update_pointcloud(widget: OpenGLWidget, points):
        print(f"Load point Cloud size: {points.shape}")
        ## updates
        widget.point_selector.update_pointcloud(points)
        widget.update_point_clouds()
        widget.repaint()

    def interpolate_pointclouds(self):

        assert self.noises is not None
        selection = self.main_opengl_widget.point_selector.pointcloud_selected

        ## part-wise interplolation

        # new
        use_latent = self.sidebar_main_subwidgets[4].isChecked()
        if self.direction_checkbox.isChecked():
            new_points = self.point_gen.interpolation_noise(z1=self.noises[0], z2=self.noises[1], selection=selection,
                                                            alpha=self.alpha, use_latent=use_latent)
            MainInterface.update_pointcloud(self.main_opengl_widget, new_points[0])
            self.save_interpolation_points(self.alpha, new_points[0])
        else:
            new_points = self.point_gen.interpolation_noise(z1=self.noises[1], z2=self.noises[0], selection=selection,
                                                            alpha=self.alpha, use_latent=use_latent)
            MainInterface.update_pointcloud(self.another_opengl_widget, new_points[0])
            self.save_interpolation_points(self.alpha, new_points[0])

        np.save(os.path.join(self.debugger.file_path(self.get_save_folder_name()),
                             f'noise_{self.interpolation_cnt}_{self.alpha}.npy'),
                self.noises[0] * (1 - self.alpha) + self.noises[1] * self.alpha)

        direction_index = 0 if self.direction_checkbox.isChecked() else 1
        self.current_noise = np.copy(self.noises[direction_index])
        self.current_noise[selection == 1, :] = self.current_noise[selection == 1, :] * (1 - self.alpha) + self.noises[
            1 - direction_index, selection == 1] * self.alpha

    def update_noise(self):
        assert self.current_noise is not None
        direction_index = 0 if self.direction_checkbox.isChecked() else 1

        self.noises[direction_index] = self.current_noise
        new_points = self.point_gen.generate_points_from_noise(self.noises[direction_index])

        ## widgets update
        widgets = [self.main_opengl_widget, self.another_opengl_widget]
        MainInterface.update_pointcloud(widgets[direction_index], new_points[0])

        ## update the index
        self.points[direction_index] = new_points[0]
        self.pointcloud_indices[direction_index] = self.pointcloud_cnt
        self.pointcloud_cnt += 1

    def save_current_selection_noise(self):
        direction_index = 0 if self.direction_checkbox.isChecked() else 1

        ## widgets
        widgets = [self.main_opengl_widget, self.another_opengl_widget]
        selection = widgets[direction_index].point_selector.pointcloud_selected
        selection_flag = selection == 1

        if np.any(selection_flag):
            ## retrived noise
            retrieved_noises = np.copy(self.noises[direction_index])[selection_flag]
            retrieved_noise = np.mean(retrieved_noises, axis=0)

            ## save the noise
            np.save(self.debugger.file_path(
                f'model_{self.pointcloud_indices[direction_index]}_partial_{self.partial_noise_index}_noise.npy'),
                    retrieved_noise)
            np.save(self.debugger.file_path(
                f'model_{self.pointcloud_indices[direction_index]}_partial_{self.partial_noise_index}_selection.npy'),
                    selection)
            self.partial_noise_index += 1
        else:
            print("No selections for points!!!")

    def init_sidebar(self):

        # functions
        self.init_sidebar_functions()

        ## layouts
        sidebar_main_layout = QVBoxLayout()
        for item in self.sidebar_main_subwidgets:
            sidebar_main_layout.addWidget(item)
        sidebar_main_layout.addStretch(10)
        sidebar_main_layout.setSpacing(20)
        sidebarMainWidget = QWidget()
        sidebarMainWidget.setLayout(sidebar_main_layout)

        sidebar_control_layout = QVBoxLayout()
        for item in self.sidebar_control_subwidgets:
            sidebar_control_layout.addWidget(item)
        sidebar_control_layout.addStretch(10)
        sidebar_control_layout.setSpacing(20)
        sidebarControlWidget = QWidget()
        sidebarControlWidget.setLayout(sidebar_control_layout)

        ## add tabs
        self.sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()
        tabs = QTabWidget()
        tabs.addTab(sidebarMainWidget, "Main")
        tabs.addTab(sidebarControlWidget, "Control")
        sidebar_layout.addWidget(tabs)
        self.sidebar_widget.setLayout(sidebar_layout)

    @staticmethod
    def reset_selection(widget: OpenGLWidget):
        widget.point_selector.reset_selection()
        widget.update_selected_points()
        widget.repaint()

    def unselect_all(self):
        MainInterface.reset_selection(self.main_opengl_widget)
        if self.another_opengl_widget is not None:
            MainInterface.reset_selection(self.another_opengl_widget)

    def update_alpha(self, value):
        print(f"Current alpha {self.alpha}")
        self.alpha = value / self.MAX_ALPHA
        self.sidebar_main_subwidgets[2].setText(str(self.alpha))
        self.interpolate_pointclouds()

    @staticmethod
    def copy_selection(widget_source: OpenGLWidget, widget_dest: OpenGLWidget):
        widget_dest.point_selector.assign_selection(widget_source.point_selector.pointcloud_selected)
        widget_dest.update_selected_points()
        widget_dest.repaint()

    def save_basic_points_info(self):
        ## name
        folder_path = self.debugger.file_path(self.get_save_folder_name())

        ## makedir if not exist
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        ## save noise
        # np.save(os.path.join(folder_path, f'noise_{self.current_folder_name.split("_")[1]}.npy'), self.noises[0])

        ## save the xyz files
        # points_0 = pc_normalize(self.points[0])

        # np.savetxt(os.path.join(folder_path, f'model_{self.current_folder_name.split("_")[1]}.xyz'), points_0)

    def save_interpolation_points(self, alpha, points):
        ## name
        folder_path = self.debugger.file_path(self.get_save_folder_name())

        ## makedir if not exist
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        ## make the color at [4:6]
        selection = np.expand_dims(self.main_opengl_widget.point_selector.pointcloud_selected, axis=1)
        points = pc_normalize(points)
        points_output = np.concatenate((points, selection), axis=1)

        text_out = 'left_to_right' if self.direction_checkbox.isChecked() else 'right_to_left'
        np.savetxt(os.path.join(folder_path, f'interpolation_{text_out}_{self.interpolation_cnt}_{alpha}.xyz'),
                   points_output)
        np.save(os.path.join(folder_path, f'selection_{text_out}_{self.interpolation_cnt}_{alpha}.npy'),
                self.main_opengl_widget.point_selector.pointcloud_selected)

        self.interpolation_cnt += 1

    def get_save_folder_name(self):
        return self.current_folder_name

    def load_noise(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        load_noise_path = r"E:\backup_pointgen\animal_segmentation_2\Animals_set\testing_shapes"
        fileName, _ = QFileDialog.getOpenFileName(self, "Open a Noise Vector", directory=load_noise_path,
                                                  initialFilter="npy file (*.npy)", options=options)
        if len(fileName) > 0 and fileName is not None:
            print(f"Open : {fileName}")
            try:
                if self.load_checkbox.isChecked():
                    # self.noises[0] = trimesh.load(fileName)
                    # self.points[0] = self.point_gen.generate_points_from_noise(self.noises[0])[0]
                    if fileName.endswith(".ply"):
                        my_point_cloud = trimesh.load(fileName).vertices
                        #np.random.shuffle(my_point_cloud)
                        self.points[0] = my_point_cloud[:2048]
                    else:
                        print("I am here")
                        my_mesh = trimesh.load(fileName)
                        my_mesh = scale_to_unit_cube(my_mesh, scale_ratio=0.45)
                        self.my_mesh = my_mesh
                        my_point_cloud = np.array(trimesh.sample.sample_surface(my_mesh, 2048)[0])
                        self.my_edit_pc = my_point_cloud
                        print (my_point_cloud.shape, my_point_cloud)
                        self.points[0] = my_point_cloud[:2048]

                    # np.random.shuffle(self.points[0])
                    # self.points[0] = self.points[0]/10
                    # self.update_pointcloud(self.main_opengl_widget, self.points[0])
                    self.update_pointcloud(self.main_opengl_widget, my_point_cloud)

                    ## generate new folder
                    self.pointcloud_indices[0] = self.pointcloud_cnt
                else:
                    self.noises[1] = np.load(fileName)
                    self.points[1] = self.point_gen.generate_points_from_noise(self.noises[1])[0]
                    self.update_pointcloud(self.another_opengl_widget, self.points[1])
                    self.pointcloud_indices[1] = self.pointcloud_cnt

                self.current_folder_name = os.path.basename(fileName).split('.')[0]
                self.save_basic_points_info()
                self.pointcloud_cnt += 1
                self.interpolation_cnt = 0
            except:
                traceback.print_exc()

    def load_selection(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        fileName, _ = QFileDialog.getOpenFileName(self, "Open a Selection File", directory=self.debugger.file_path('.'),
                                                  initialFilter="npy file (*.npy)", options=options)
        if len(fileName) > 0 and fileName is not None:
            print(f"Open : {fileName}")
            try:
                selection = np.load(fileName)
                self.main_opengl_widget.point_selector.assign_selection(selection)
                self.main_opengl_widget.update_selected_points()
                self.main_opengl_widget.repaint()

                if self.use_two_screen:
                    self.another_opengl_widget.point_selector.assign_selection(selection)
                    self.another_opengl_widget.update_selected_points()
                    self.another_opengl_widget.repaint()
            except:
                traceback.print_exc()


    def save_head_selection(self):
        self.save_selection(index=1)

    def save_tail_selection(self):
        self.save_selection(index=0)

    def save_feet_selection(self):
        self.save_selection(index=2)

    def save_body_selection(self):
        self.save_selection(index=3)

    def get_partial_mesh(self, mesh, partial_pc):
        mesh_vertices = mesh.vertices


    def save_selection(self, index: int):
        selection = self.main_opengl_widget.point_selector.pointcloud_selected
        current_folder = self.get_save_folder_name()
        # np.save(self.debugger.file_path(os.path.join(current_folder, f'selection.npy')), selection)
        if not os.path.exists(os.path.join(self.debugger.pre_fix, current_folder)):
            os.mkdir(os.path.join(self.debugger.pre_fix, current_folder))
        np.save(os.path.join(self.debugger.pre_fix, current_folder, "selection.npy"), selection)
        # np.save((self.debugger.file_path(os.path.join(current_folder, f'selection_{index}.npy')), self.points[0]))
        # np.savetxt(self.debugger.file_path(os.path.join(current_folder, f"full_pc.xyz")), self.my_edit_pc)
        #print (self.my_edit_pc, 'self.my_edit_pc')
        #np.savetxt(os.path.join(self.debugger.pre_fix, current_folder, "full_pc.xyz"), self.my_edit_pc)
        self.selection_cnt += 1

    def add_selection(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        fileName, _ = QFileDialog.getOpenFileName(self, "Open a Selection File", directory=self.debugger.file_path('.'),
                                                  initialFilter="npy file (*.npy)", options=options)
        if len(fileName) > 0 and fileName is not None:
            print(f"Open : {fileName}")
            try:
                selection = np.load(fileName)
                selection = np.clip(self.main_opengl_widget.point_selector.pointcloud_selected + selection, 0, 1)
                self.main_opengl_widget.point_selector.assign_selection(selection)
                self.main_opengl_widget.update_selected_points()
                self.main_opengl_widget.repaint()

                if self.use_two_screen:
                    self.another_opengl_widget.point_selector.assign_selection(selection)
                    self.another_opengl_widget.update_selected_points()
                    self.another_opengl_widget.repaint()

            except:
                traceback.print_exc()

    def load_noise_on_selection(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        fileName, _ = QFileDialog.getOpenFileName(self, "Open a Partial Noise File",
                                                  directory=self.debugger.file_path('.'),
                                                  initialFilter="npy file (*.npy)", options=options)
        if len(fileName) > 0 and fileName is not None:
            print(f"Open : {fileName}")
            try:
                new_noise_loaded = np.load(fileName)
                assert len(new_noise_loaded.shape) == 1
                selection = self.main_opengl_widget.point_selector.pointcloud_selected

                ## update the main point cloud
                new_noises = np.copy(self.noises[0])
                new_noises[selection == 1] = new_noise_loaded
                new_point = self.point_gen.generate_points_from_noise(new_noises)[0]
                MainInterface.update_pointcloud(self.main_opengl_widget, new_point)
                self.main_opengl_widget.repaint()

                ## update indices
                self.noises[0] = new_noises
                self.points[0] = new_point
                self.pointcloud_indices[0] = self.pointcloud_cnt
                self.pointcloud_cnt += 1

                ## udpadte another poitcloud
                if self.another_opengl_widget is not None:
                    new_noises_2 = np.tile(np.expand_dims(new_noise_loaded, axis=0), (new_noises.shape[0], 1))
                    new_point_2 = self.point_gen.generate_points_from_noise(new_noises_2)[0]
                    MainInterface.update_pointcloud(self.another_opengl_widget, new_point_2)
                    self.another_opengl_widget.repaint()

                    self.noises[1] = new_noises_2
                    self.points[1] = new_point_2
                    self.pointcloud_indices[1] = self.pointcloud_cnt
                    self.pointcloud_cnt += 1



            except:
                traceback.print_exc()

    def toggle_selection(self):

        selection = self.main_opengl_widget.point_selector.pointcloud_selected
        reflect_selection = 1 - selection

        self.main_opengl_widget.point_selector.assign_selection(reflect_selection)
        self.main_opengl_widget.update_selected_points()
        self.main_opengl_widget.repaint()

        if self.use_two_screen:
            self.another_opengl_widget.point_selector.assign_selection(reflect_selection)
            self.another_opengl_widget.update_selected_points()
            self.another_opengl_widget.repaint()

    def update_light_position(self, value):
        self.pointLight_angle = value / self.MAX_ALPHA * np.pi * 2
        self.main_opengl_widget.current_light_angle = self.pointLight_angle
        self.main_opengl_widget.updateLightPosition()
        self.main_opengl_widget.repaint()
        if self.another_opengl_widget is not None:
            self.another_opengl_widget.current_light_angle = self.pointLight_angle
            self.another_opengl_widget.updateLightPosition()
            self.another_opengl_widget.repaint()

    def update_FOV(self, value):
        self.main_opengl_widget.point_selector.currFovy = value
        self.main_opengl_widget.point_selector.reset_projection()
        self.main_opengl_widget.repaint()
        if self.another_opengl_widget is not None:
            self.another_opengl_widget.point_selector.currFovy = value
            self.another_opengl_widget.point_selector.reset_projection()
            self.another_opengl_widget.repaint()

    def update_selected_color(self):
        current_selection_rgb = self.main_opengl_widget.selected_color
        current_color = QColor(current_selection_rgb[0], current_selection_rgb[1], current_selection_rgb[2])
        color = QColorDialog.getColor(current_color)
        if color.isValid():
            new_color = np.array((color.red(), color.green(), color.blue()), dtype=np.float32)
            self.main_opengl_widget.selected_color = new_color
            self.main_opengl_widget.update_Selection_Color()
            if self.another_opengl_widget is not None:
                self.another_opengl_widget.selected_color = new_color
                self.another_opengl_widget.update_Selection_Color()

    def update_unselected_color(self):
        current_unselected_rgb = self.main_opengl_widget.unselected_color
        current_color = QColor(current_unselected_rgb[0], current_unselected_rgb[1], current_unselected_rgb[2])
        color = QColorDialog.getColor(current_color)
        if color.isValid():
            new_color = np.array((color.red(), color.green(), color.blue()), dtype=np.float32)
            self.main_opengl_widget.unselected_color = new_color
            self.main_opengl_widget.update_Selection_Color()
            if self.another_opengl_widget is not None:
                self.another_opengl_widget.unselected_color = new_color
                self.another_opengl_widget.update_Selection_Color()

    def updateBallSize(self, value):
        print("current ball size :", value / self.BALL_RATIO)
        self.main_opengl_widget.point_selector.current_ball_size = value / self.BALL_RATIO
        self.main_opengl_widget.point_selector.update_ball()
        self.main_opengl_widget.update_point_clouds()
        self.main_opengl_widget.repaint()
        if self.another_opengl_widget is not None:
            self.another_opengl_widget.point_selector.current_ball_size = value / self.BALL_RATIO
            self.another_opengl_widget.point_selector.update_ball()
            self.another_opengl_widget.update_point_clouds()
            self.another_opengl_widget.repaint()

    def save_for_gallery(self):
        PER_IMAGE = 15
        only_right = self.use_two_screen and not self.load_checkbox.isChecked()
        point_index = 1 if only_right else 0

        models_path = debugger.file_path(f'models_{self.pointcloud_indices[point_index] // PER_IMAGE}')

        selection_path = debugger.file_path('selections')
        ## get save thing
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
        if not os.path.isdir(selection_path):
            os.mkdir(selection_path)

        ## save the thing
        points = self.points[point_index]
        noises = self.noises[point_index]
        selection = np.expand_dims(self.main_opengl_widget.point_selector.pointcloud_selected, axis=1)
        points_output = np.concatenate((points, selection), axis=1)

        np.save(os.path.join(selection_path, f'noises_{self.pointcloud_indices[point_index]}.npy'), noises)
        np.savetxt(os.path.join(models_path, f'points_{self.pointcloud_indices[point_index]}.xyz'), points_output)

    def update_noise_search_dir(self):
        self.current_noise_dir = np.random.rand(self.point_gen.model.opts.nz) * 2 - 1
        self.current_noise_dir = self.current_noise_dir / np.linalg.norm(self.current_noise_dir)
        if hasattr(self, 'noise_slider'):
            self.noise_slider.setValue(0)

    def update_noise_search_result(self):
        change_index = 0 if self.direction_checkbox.isChecked() else 1
        selections = self.main_opengl_widget.point_selector.pointcloud_selected if self.direction_checkbox.isChecked() else self.another_opengl_widget.point_selector.pointcloud_selected
        new_noise = np.copy(self.noises[change_index])
        new_noise[selections == 1] = self.current_noise_dir[np.newaxis, :] * self.current_noise_dir_ratio + new_noise[
            selections == 1]
        new_points = self.point_gen.generate_points_from_noise(new_noise)

        self.permanant_update_noise(change_index, new_noise, new_points)

        self.noise_slider.setValue(0)

    def permanant_update_noise(self, change_index, new_noise, new_points):
        ## update noises and points
        self.noises[change_index] = new_noise
        self.points[change_index] = new_points[0]
        ## update the picture as well
        widgets = [self.main_opengl_widget, self.another_opengl_widget]
        MainInterface.update_pointcloud(widgets[change_index], new_points[0])
        ## update the index
        self.pointcloud_indices[change_index] = self.pointcloud_cnt
        self.pointcloud_cnt += 1

    def change_model_dir(self, value):
        change_index = 0 if self.direction_checkbox.isChecked() else 1
        selections = self.main_opengl_widget.point_selector.pointcloud_selected if self.direction_checkbox.isChecked() else self.another_opengl_widget.point_selector.pointcloud_selected
        new_noise = np.copy(self.noises[change_index])
        new_noise[selections == 1] = self.current_noise_dir[np.newaxis, :] * (
                    value / self.MAX_SCALE_NOISE) * self.MAX_NOISE_RATIO + new_noise[selections == 1]
        new_points = self.point_gen.generate_points_from_noise(new_noise)
        if change_index == 0:
            MainInterface.update_pointcloud(self.main_opengl_widget, new_points[0])
        else:
            MainInterface.update_pointcloud(self.another_opengl_widget, new_points[0])

        self.current_noise_dir_ratio = value / self.MAX_SCALE_NOISE * self.MAX_NOISE_RATIO

        print(f"current search ratio : {self.current_noise_dir_ratio}")

    def update_selection_mode(self, value):
        if self.main_opengl_widget is not None:
            if self.selection_mode_checkbox.isChecked():
                self.main_opengl_widget.point_selector.selection_mode = 1
                if self.another_opengl_widget is not None:
                    self.another_opengl_widget.point_selector.selection_mode = 1
            else:
                self.main_opengl_widget.point_selector.selection_mode = 0
                if self.another_opengl_widget is not None:
                    self.another_opengl_widget.point_selector.selection_mode = 0

    def random_update_noises(self):
        self.current_noise = np.random.normal(0, 0.2, (self.point_gen.model.opts.nz))

        ## set changing index
        direction_clicked = not hasattr(self, 'direction_checkbox') or self.direction_checkbox.isChecked()
        change_index = 0 if direction_clicked else 1
        selections = self.main_opengl_widget.point_selector.pointcloud_selected if direction_clicked else self.another_opengl_widget.point_selector.pointcloud_selected
        new_noise = np.copy(self.noises[change_index])
        new_noise[selections == 1] = self.current_noise
        new_points = self.point_gen.generate_points_from_noise(new_noise)

        self.permanant_update_noise(change_index, new_noise, new_points)


if __name__ == '__main__':
    select_chairs = True
    debugger = MyDebugger("Main_Interface", select_chairs)
    app = QApplication(sys.argv)
    ex = MainInterface(debugger=debugger, use_part_seg=False, use_two_screen=False, use_point_gen=True)

    print("Start to show the windows!!!!!")
    ex.show()
    sys.exit(app.exec_())