import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

import csv

dic={}
with open('captions.tablechair.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        ##print (row)
        #print (row[1], row[2])
        dic[row[1]]=row[2]


image = preprocess(Image.open('bunny3.png')).unsqueeze(0).to(device)
text1 = clip.tokenize('bunny').to(device)
text2 = clip.tokenize('teapot').to(device)
with torch.no_grad():
    image_features = model.encode_image(image)
    image_features=image_features/torch.norm(image_features,p=2,dim=-1)
    text_features = model.encode_text(text1)
    text_features=text_features/torch.norm(text_features,p=2,dim=-1)
    vmax=torch.sum(image_features*text_features).item()

    text_features = model.encode_text(text2)
    text_features=text_features/torch.norm(text_features,p=2,dim=-1)
    vmin=torch.sum(image_features*text_features).item()



#print (vmax,vmin)
#exit()
values=0.0
cnt=0
import glob
#print (dic.keys())
paths=glob.glob('pred/*/meshnew/meshnew_r_000.png')
#paths=glob.glob('/home/user/text/*/mesh_text/mesh_text_r_000.png')
for path in paths:
 try:
  name=path.split('/')[-3].split('_')[0]
  #print (name)
      
  image = preprocess(Image.open(path)).unsqueeze(0).to(device)
  text = clip.tokenize(dic[name]).to(device)
  
  with torch.no_grad():
      image_features = model.encode_image(image)
      image_features=image_features/torch.norm(image_features,p=2,dim=-1)
      text_features = model.encode_text(text)
      text_features=text_features/torch.norm(text_features,p=2,dim=-1)
      
      value=torch.sum(image_features*text_features).item()
      print (dic[name], value)
      
      if value>vmax:
        value=1
      elif value<vmin:
        value=0
      else:
        value=(value-vmin)/(vmax-vmin)
      
      

      #print (value)
      values+=value
      cnt+=1
      print (values, cnt, values/cnt)
 except:
    pass
      
print (values/cnt)
