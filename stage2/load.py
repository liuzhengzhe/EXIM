import numpy as np
import glob,os
paths=glob.glob('shapenet/data//03001627/*/boundary_0.1_samples.npz')
for path in paths:
  data=os.path.getsize(path)
  if data==0:
    print (path,data)
    os.remove(path)