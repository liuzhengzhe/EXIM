import os
import math
opts = lux.getImportOptions()
print(opts)


opts['accurate_tessellation'] = False
opts['adjust_camera_look_at'] =  False
opts['adjust_environment'] =  True
opts['applyLibraryMaterials'] =  False
opts['camera_import'] =  True
opts['center_geometry'] =  True
opts['compute_normals'] =  True
opts['frame'] =  0
opts['geometry_scale'] =  10
opts['geometry_units'] =  1000.0
opts['group_by'] =  2
opts['group_by_shader'] =  False
opts['include_hidden_surfaces'] =  False
opts['include_nurbs'] =  False
opts['include_single_surfaces'] =  True
opts['material_name_from_color'] =  False
opts['mayaForceVersion'] =  ''
opts['merge_groups'] =  False
opts['merge_objects'] =  False
opts['new_import'] =  False
opts['retain_materials'] =  True
opts['same_coordinates'] =  True
opts['separate_materials'] =  True
opts['separate_parts'] =  True
opts['snap_to_ground'] =  True
opts['tessellation_quality'] =  0.20000000298023224
opts['up_vector'] =  1
opts['update_mode'] =  False


#cwd = r'F:\mesh'
#cwd = r'D:\render\baseline\nogan\mesh_new' #r'D:\render\mesh_new\old'
#cwd = r'D:\render\0503\mesh_new'
#cwd = r'D:\render\cabinet\cabinet\mesh_new'
#cwd=r'D:\render\table\table\mesh_new'
#cwd=r'D:\render\sofa\sofa\mesh_new'
#cwd=r'D:\0815'
cwd=r'C:\Users\liuzh\Desktop\0827\0827\2'
#cwd=r'D:\render\chandelier\chandelier\mesh_new'
batch_size = 1
print(cwd) # print current working directory

file_names = []
for root, dirs, files in os.walk(cwd, topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        if file_path.endswith('.obj'):
            file_names.append(file_path)


batch_num = int(math.ceil(len(file_names)))


# get material
material = None
root = lux.getSceneTree()
#for node in root.find( types = lux.NODE_TYPE_MODEL ):
#    if node.getName() == 'material_base':
#        material = node.getMaterial()

#assert material is not None

for i in range(batch_num):
    print (i)
    current_files = file_names[i*batch_size:(i+1)*batch_size]

    ## load and set material
    for file_path in current_files:
        node = lux.importFile(file_path, opts=opts)

    ## Render
    #for node in root.find(types=lux.NODE_TYPE_MODEL):
    #    node.setMaterial(material)
    #    node.hide()

    for idx, node in enumerate(root.find(types=lux.NODE_TYPE_MODEL)):
        if node.getName() != 'material_base':
            #node.show()
            lux.renderImage(os.path.join(cwd, "image_"  + node.getName() + ".png"), width=1024, height=1024)
            #node.hide()

    ## Remove node
    lux.clearGeometry()





