import glob
import os
from PIL import Image

os.makedirs(r'dataset\frames\right_punch_2', exist_ok=True)
os.makedirs(r'dataset\frames\left_punch_2', exist_ok=True)

right_punch_counter = 2285
left_punch_counter = 2313




counter = left_punch_counter
dir = 'left_punch'
files = glob.glob(r'dataset\frames\both_punches\*')

counters = [2313, 2285]
dirs = ['left_punch_2', 'right_punch_2']
indices = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1] * ((len(files) // 10) // 2)
indices += [0,0,0,0,0,0,0,0,0,0]

for idx, file in enumerate(files):
    index = indices[idx]
    img = Image.open(file)
    img_name = 'f-0{}.jpg'.format(counters[index])
    #print(os.path.join(r'dataset\frames', dirs[index], img_name))
    img.save(os.path.join(r'dataset\frames', dirs[index], img_name))
    counters[index] += 1
    
    
    


