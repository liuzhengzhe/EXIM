from PyQt5.QtGui import  *
from PyQt5.QtWidgets import QOpenGLWidget
from PointSelector.PointSelector import PointSelector, MOUSE_RIGHT_BUTTON
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from UI.shaders.pointclouds_vertex_shader import pointclouds_vertex_shader
from UI.shaders.pointclouds_fragment_shader import pointclouds_fragment_shader
from UI.shaders.pointclouds_balls_vertex_shader import pointclouds_balls_vertex_shader
from UI.shaders.pointclouds_balls_fragment_shader import pointclouds_balls_fragment_shader
from UI.shaders.polygon_vertex_shader import  polygon_vertex_shader
from UI.shaders.polygon_fragment_shader import polygon_fragment_shader
import numpy as np
import ctypes
import glm

## init Opengl version
format = QSurfaceFormat()
format.setVersion(4, 3)
profile = QSurfaceFormat.CoreProfile
format.setProfile(profile)
QSurfaceFormat.setDefaultFormat(format)
print(f"OpenGL version : {format.majorVersion()}.{format.minorVersion()}")

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None, use_part_seg = False, points = None, use_ball = True, initial_fov = 30.0, ball_size = 0.015):
        super().__init__(parent)

        ## init some openGL stuff here
        self.use_ball = use_ball
        self.pointcloud_program_id = None
        self.polygon_program_id = None
        self.VAOs = []
        self.VBOs = []
        self.EBOs = []

        ## color
        self.selected_color = np.array([0.0, 127.0, 255.0], dtype = np.float32)
        self.unselected_color = np.array([249.0, 178.0, 70.0], dtype = np.float32)

        ## init the point cloud
        self.point_selector = PointSelector(winW = self.width(), winH = self.height(), use_part_seg= use_part_seg, points = points, initial_Fov = initial_fov, ball_size = ball_size)

        ## lighting
        self.LIGHT_MAX_DISTANCE = 1.0
        self.LIGHT_MIN_DISTANCE = -1.0
        self.HEIGHT = 1.0
        self.current_light_angle = 0.0
        self.light_Position = None
        self.updateLightPosition()


    def initializeGL(self) -> None:

        ## initil
        if self.use_ball:
            glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)

        ## background color
        # glClearColor(0.5, 0.8, 0.7, 1.0)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        ## init the shader
        self.init_shader()

        ## send the pointcloud
        self.define_point_clouds()
        self.define_selection_polygon()

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)
        self.point_selector.windowW = self.width()
        self.point_selector.windowH = self.height()
        self.point_selector.reset_projection()
        self.repaint()

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_DEPTH_BUFFER_BIT)

        ## use
        glUseProgram(self.polygon_program_id)
        self.draw_polygon()

        ## use the shader for pointcloud
        glUseProgram(self.pointcloud_program_id)
        self.draw_pointclouds()


    def draw_pointclouds(self):
        ## draw the pointcloud
        glBindVertexArray(self.VAOs[0])

        ## set two matrix
        modelViewMatrix_loc = glGetUniformLocation(self.pointcloud_program_id, "modelViewMatrix")
        projectionMatrix_loc = glGetUniformLocation(self.pointcloud_program_id, "projectionMatrix")

        glUniformMatrix4fv(modelViewMatrix_loc, 1, GL_FALSE, np.array(self.point_selector.modelview_matrix))
        glUniformMatrix4fv(projectionMatrix_loc, 1, GL_FALSE, np.array(self.point_selector.projection_matrix))


        if not self.use_ball:
            glDrawArrays(GL_POINTS, 0, self.point_selector.pointcloud_arr.shape[0])
        else:
            ## get the eyes loc
            eyePosition_loc = glGetUniformLocation(self.pointcloud_program_id, "eyePosition")
            view_matrix_inverse = glm.inverse(self.point_selector.modelview_matrix)
            eyePosition = np.array([view_matrix_inverse[3, 0], view_matrix_inverse[3, 1], view_matrix_inverse[3, 2]],
                                   dtype=np.float32)
            glUniform3fv(eyePosition_loc, 1, eyePosition)

            ## lighting
            pointLightPosition_loc = glGetUniformLocation(self.pointcloud_program_id, "pointLightPosition")
            glUniform3fv(pointLightPosition_loc, 1, self.light_Position)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBOs[0])
            indices_to_draw = self.point_selector.pointcloud_arr.shape[0] * self.point_selector.ball_template[1].shape[0]
            glDrawElements(GL_TRIANGLES, indices_to_draw * 3, GL_UNSIGNED_INT, None)

    def draw_polygon(self):
        ## draw the polygon
        glBindVertexArray(self.VAOs[1])
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[1])

        ## set the polygon
        polygon_points = np.array(self.point_selector.selection_polygon, dtype = np.float32)
        glBufferData(GL_ARRAY_BUFFER, polygon_points.nbytes,
                     polygon_points, GL_STATIC_DRAW)


        # Draw
        if self.point_selector.pressed_mouse_button == MOUSE_RIGHT_BUTTON:
            glDrawArrays(GL_LINE_STRIP, 0, polygon_points.shape[0])
        else:
            glDrawArrays(GL_LINE_LOOP, 0, polygon_points.shape[0])

    def define_point_clouds(self):


        if not self.use_ball:
            ## functions for sending the definition of pointclouds
            # get VAO
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            ## vertex buffer data
            vbo_1 = glGenBuffers(1)
            point_arrs = self.point_selector.pointcloud_arr
            glBindBuffer(GL_ARRAY_BUFFER, vbo_1)
            glBufferData(GL_ARRAY_BUFFER, point_arrs.nbytes,
                         point_arrs, GL_STATIC_DRAW)

            ## get Buffer the data
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            ## Blind the information
            vbo_2 = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_2)

            ## set attribute Pointer
            # loc = glGetAttribLocation(self.pointcloud_program_id, "selected")
            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)
            points_selection = self.point_selector.pointcloud_selected
            glBufferData(GL_ARRAY_BUFFER, points_selection.nbytes,
                         points_selection, GL_DYNAMIC_DRAW)

            ## Blind the information
            vbo_3 = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_3)
            segmentation = self.point_selector.points_segmentation
            glBufferData(GL_ARRAY_BUFFER, segmentation.nbytes,
                         segmentation, GL_DYNAMIC_DRAW)

            # loc = glGetAttribLocation(self.pointcloud_program_id, "segmented")
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)

            ## update the vao and vbo
            self.VAOs.append(vao)
            self.VBOs.append((vbo_1, vbo_2))
        else:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            ## vertex buffer data position of the ball
            vbo_1 = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_1)
            point_arrs = self.point_selector.pointcloud_arr
            updated_points_arrs = point_arrs[:, np.newaxis, :] + self.point_selector.ball_template[0][np.newaxis, :, :]
            updated_points_arrs = np.reshape(updated_points_arrs, (-1, 3))
            glBufferData(GL_ARRAY_BUFFER, updated_points_arrs.nbytes,
                         updated_points_arrs, GL_STATIC_DRAW)

            ## get Buffer the data
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            ## Blind the information
            vbo_2 = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_2)
            points_selection = self.point_selector.pointcloud_selected
            updated_points_selection = np.tile(np.expand_dims(points_selection, axis = 1), (1, self.point_selector.ball_template[0].shape[0])) ## expand T times
            updated_points_selection = np.reshape(updated_points_selection, -1)
            glBufferData(GL_ARRAY_BUFFER, updated_points_selection.nbytes,
                         updated_points_selection, GL_DYNAMIC_DRAW)

            glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

            ## Blind the information
            vbo_3 = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_3)
            segmentation = self.point_selector.points_segmentation
            updated_segmentation = np.tile(np.expand_dims(segmentation, axis = 1), (1, self.point_selector.ball_template[0].shape[0])) ## expand T times
            updated_segmentation = np.reshape(updated_segmentation, -1)
            glBufferData(GL_ARRAY_BUFFER, updated_segmentation.nbytes,
                         updated_segmentation, GL_DYNAMIC_DRAW)

            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)

            ## BUFFER once is fine for the normal becaue we only have rigid transform
            vbo_4 = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_4)
            ball_normals = self.point_selector.ball_template[2]
            updated_ball_normals = np.tile(np.expand_dims(ball_normals, axis = 0), (self.point_selector.pointcloud_arr.shape[0], 1, 1)) ## expand N times
            updated_ball_normals = np.reshape(updated_ball_normals, (-1, 3))
            glBufferData(GL_ARRAY_BUFFER, updated_ball_normals.nbytes,
                         updated_ball_normals, GL_DYNAMIC_DRAW)

            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(3)

            ## LASTLY : THE INDEX never need to change this --> buffer once is okay
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            indices = self.point_selector.ball_template[1]
            stepping_array = np.arange(self.point_selector.pointcloud_arr.shape[0]) * self.point_selector.ball_template[0].shape[0]
            indices_updated = indices[np.newaxis, :, :] + stepping_array[:, np.newaxis, np.newaxis] ## calculate index
            indices_updated = np.reshape(indices_updated, (-1, 3))

            ## convert to unsigned int
            indices_updated = np.array(indices_updated, dtype = np.uint32)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_updated.nbytes,
                         indices_updated, GL_STATIC_DRAW)

            self.VAOs.append(vao)
            self.VBOs.append((vbo_1, vbo_2, vbo_3))
            self.EBOs.append(ebo)

            ## update color
            self.update_Selection_Color()

    def update_Selection_Color(self):
        ## set color
        glUseProgram(self.pointcloud_program_id)
        selectedColor_loc = glGetUniformLocation(self.pointcloud_program_id, "selectedColor")
        unselectedColor_loc = glGetUniformLocation(self.pointcloud_program_id, "unselectedColor")

        assert selectedColor_loc != -1 and unselectedColor_loc != -1

        glUniform3fv(selectedColor_loc, 1, self.selected_color)
        glUniform3fv(unselectedColor_loc, 1, self.unselected_color)

    def define_selection_polygon(self):

        ## set the VBO
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)

        ## functions for sending the definition of pointclouds
        # get VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        loc = glGetAttribLocation(self.polygon_program_id, "position")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        ## update the vao and vbo
        self.VAOs.append(vao)
        self.VBOs.append(vbo)

    def update_selected_points(self):
        ## for the selected flag
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[0][1])
        points_selection = self.point_selector.pointcloud_selected
        if not self.use_ball:
            glBufferData(GL_ARRAY_BUFFER, points_selection.nbytes,
                         points_selection, GL_DYNAMIC_DRAW)
        else:
            updated_points_selection = np.tile(np.expand_dims(points_selection, axis=1),
                                               (1, self.point_selector.ball_template[0].shape[0]))  ## expand T times
            updated_points_selection = np.reshape(updated_points_selection, -1)
            glBufferData(GL_ARRAY_BUFFER, updated_points_selection.nbytes,
                         updated_points_selection, GL_DYNAMIC_DRAW)


    def update_point_clouds(self):
        point_arrs = self.point_selector.pointcloud_arr
        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[0][0])

        if not self.use_ball:
            glBufferData(GL_ARRAY_BUFFER, point_arrs.nbytes,
                         point_arrs, GL_STATIC_DRAW)
        else:

            ## update position
            updated_points_arrs = point_arrs[:, np.newaxis, :] + self.point_selector.ball_template[0][np.newaxis, :, :]
            updated_points_arrs = np.reshape(updated_points_arrs, (-1, 3))
            glBufferData(GL_ARRAY_BUFFER, updated_points_arrs.nbytes,
                         updated_points_arrs, GL_STATIC_DRAW)

            # update segmentation
            glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[0][2])
            segmentation = self.point_selector.points_segmentation
            updated_segmentation = np.tile(np.expand_dims(segmentation, axis = 1), (1, self.point_selector.ball_template[0].shape[0])) ## expand T times
            updated_segmentation = np.reshape(updated_segmentation, -1)
            glBufferData(GL_ARRAY_BUFFER, updated_segmentation.nbytes,
                         updated_segmentation, GL_DYNAMIC_DRAW)
    ## init the shader
    def init_shader(self):

        ## pointcloud program
        self.pointcloud_program_id = glCreateProgram()
        if not self.use_ball:
            vs_id = self.add_shader(pointclouds_vertex_shader, GL_VERTEX_SHADER)
            frag_id = self.add_shader(pointclouds_fragment_shader, GL_FRAGMENT_SHADER)
        else:
            vs_id = self.add_shader(pointclouds_balls_vertex_shader, GL_VERTEX_SHADER)
            frag_id = self.add_shader(pointclouds_balls_fragment_shader, GL_FRAGMENT_SHADER)

        glAttachShader(self.pointcloud_program_id, vs_id)
        glAttachShader(self.pointcloud_program_id, frag_id)
        glLinkProgram(self.pointcloud_program_id)

        # error handling
        if glGetProgramiv(self.pointcloud_program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.pointcloud_program_id)
            glDeleteProgram(self.pointcloud_program_id)
            glDeleteShader(vs_id)
            glDeleteShader(frag_id)
            raise RuntimeError('Error linking program: %s' % (info))

        ## polygon program
        self.polygon_program_id = glCreateProgram()
        vs_id = self.add_shader(polygon_vertex_shader, GL_VERTEX_SHADER)
        frag_id = self.add_shader(polygon_fragment_shader, GL_FRAGMENT_SHADER)

        glAttachShader(self.polygon_program_id, vs_id)
        glAttachShader(self.polygon_program_id, frag_id)
        glLinkProgram(self.polygon_program_id)

        if glGetProgramiv(self.polygon_program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.polygon_program_id)
            glDeleteProgram(self.polygon_program_id)
            glDeleteShader(vs_id)
            glDeleteShader(frag_id)
            raise RuntimeError('Error linking program: %s' % (info))

    def updateLightPosition(self):
        self.light_Position = np.array((np.sin(self.current_light_angle) * (self.LIGHT_MAX_DISTANCE - self.LIGHT_MIN_DISTANCE) + self.LIGHT_MIN_DISTANCE,
                                        self.HEIGHT,
                                        np.cos(self.current_light_angle) * (self.LIGHT_MAX_DISTANCE - self.LIGHT_MIN_DISTANCE) + self.LIGHT_MIN_DISTANCE), dtype = np.float32)


    def add_shader(self, source, shader_type):
        """ Helper function for compiling a GLSL shader
        Parameters
        ----------
        source : str
            String containing shader source code
        shader_type : valid OpenGL shader type
            Type of shader to compile
        Returns
        -------
        value : int
            Identifier for shader if compilation is successful
        """

        try:
            shader_id = glCreateShader(shader_type)
            glShaderSource(shader_id, source)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            else:
                print("Successfully compile the shader!")
                return shader_id
        except:
            glDeleteShader(shader_id)
            raise
