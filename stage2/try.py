import open3d as o3d
import numpy as np
mesh = o3d.io.read_triangle_mesh('mani/it has thick backrest1.npy_it is yellow in color.obj')



colors=np.asarray(mesh.vertex_colors)
print (colors.shape)