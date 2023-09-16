import numpy as np
dic=np.load('../dic.npy',allow_pickle=1)[()]
dicnew={}

colors=['red', 'yellow', 'green', 'blue', 'black', 'white', 'silver', 'purple', 'pink', 'bronze', 'wooden', 'glass', 'dark', 'metal','orange']

colors_dic={}
for color in colors:
  colors_dic[color]=0

for key in dic.keys():
  texts=dic[key]
  for text in texts:
    for color in colors:
      if color in text:

        if colors_dic[color]>500:
          continue
        colors_dic[color]+=1
        if key not in dicnew.keys():
          dicnew[key]=[text]
        else:
          dicnew[key].append(text)
        print (text)

print (colors_dic)
np.save('balance.npy', dicnew)