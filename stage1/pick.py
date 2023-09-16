import glob,shutil,os
import numpy as np
paths=glob.glob('feat_evaluation/*')
dic=np.load('../chair_test.npy',allow_pickle=1)[()]
for path in paths:
  name=path.split('/')[-1].split('_')[0]
  text=path.split('/')[-1].split('_')[1][:10]
  if name not in dic.keys():
    continue
  text_gt=dic[name][0][:10]
  if text==text_gt:
    print (text, text_gt)
    os.rename(path, path.replace('feat_evaluation','feat_evaluation_1'))
  