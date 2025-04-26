import os
import glob

img_files = glob.glob(r'dataset\frames_16\left_tilt\input_imgs\*')
for img_file in img_files:
    _, img_name = os.path.split(img_file)
    
    prefix_ascii = ord(img_name[0])
    if prefix_ascii > 99:
        os.remove(img_file)
    
    