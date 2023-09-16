import glob,os
paths=glob.glob('mesh/*')
for path in paths:
  print (path)
  name=path.split('/')[-1]
  name=name.replace('low.npy.obj','_265.obj')
  os.rename(path,'mesh/'+name)