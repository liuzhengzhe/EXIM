import glob,random
import numpy as np
f=open('filter.txt', 'w')
paths=glob.glob('../data/04379243_train/*')
for path in paths:
  name=path.split('/')[-1].split('.')[0]
  data=np.load(path)
  whe=np.where(data<0)[0]
  #print (path, whe.shape[0])
  if whe.shape[0]>22000:
    if random.randint(0,3)>0:
      f.write(name+'\n')
f.close()
    