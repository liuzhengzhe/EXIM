import numpy as np
import json
f = open('random_captions/chair.json')
data=json.load(f)
print (len(data))
for text in data[-500:]:
  print (text)