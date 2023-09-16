import requests
from PIL import Image

#url = 'https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg'
#image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
#display(image.resize((596, 437)))

#image = Image.open('/mnt/sdc/lzz/ShapeNet/02691156/10155655850468db78d106ce0a280f87/image/0001.png').convert('RGB')  


from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")



model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)


device = "cuda" # if torch.cuda.is_available() else "cpu"
model.to(device)




'''generated_ids = model.generate(**inputs, max_new_tokens=40)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)'''


'''prompt = "this is an airplane that"

inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float32)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)'''

import glob

category='03001627'

paths=glob.glob('/mnt/sdc/lzz/ShapeNet/'+category+'/*/img_choy2016/000.jpg')


paths.sort()
f=open(category+'.txt', 'w')
for path in paths:
  name=path.split('/')[-3]
  
  image = Image.open(path).convert('RGB')  
  
  inputs = processor(image, return_tensors="pt")
  
  
  prompt = "Question: Please describe the chair as detail as possible, including its shape,  color, structure, and attributes. Answer:"
  
  inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float32)
  
  generated_ids = model.generate(**inputs, max_new_tokens=77)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
  
  print(name, generated_text)
  
  f.write(name+'___'+generated_text+'\n')

f.close()
  
  