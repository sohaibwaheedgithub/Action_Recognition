import os
import cv2
import glob
import constants
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.ndimage.interpolation import zoom





def zoom_images(dir):
    img_files = glob.glob(os.path.join(dir, 'input_imgs', '*'))
    img_files = np.array(img_files)
    _, action = os.path.split(dir)

    print(f"Processing Frames of {action.title()}")

    img_root, img_name = os.path.split(img_files[-1])
    img_no = int(img_name[2:-4])
    img_prefix = chr(ord(img_name[0]) + 1)
    img_no = 0
    print(len(img_files))

    
    
    
    
    x1 = 30
    x2 = 460
    y1 = 130
    y2 = 500
    for img_file in img_files:
        img_no += 1
        new_img_name = "{}-{:05d}.jpg".format(img_prefix, img_no)
        img = plt.imread(img_file)


        img = img[x1:x2, y1:y2]
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img.save(os.path.join(dir, 'input_imgs', new_img_name))
        
        
        
        
        

        
    zout = 150
    for img_file in img_files:
        img_no += 1
        new_img_name = "{}-{:05d}.jpg".format(img_prefix, img_no)
        img = plt.imread(img_file)
            

        img = cv2.copyMakeBorder(img, zout, zout, zout, zout, cv2.BORDER_CONSTANT, (0,0,0))
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img.save(os.path.join(dir, 'input_imgs', new_img_name))
        



    
    x1 = 100 #30
    x2 = 430 # 470
    y1 = 0#100
    y2 = -1 #540
    for img_file in img_files:
        img_no += 1
        new_img_name = "{}-{:05d}.jpg".format(img_prefix, img_no)
        img = plt.imread(img_file)


        img = img[x1:x2, y1:y2]
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img.save(os.path.join(dir, 'input_imgs', new_img_name))
        
        


    x1 = 0 
    x2 = -1 
    y1 = 100
    y2 = 540 
    for img_file in img_files:
        img_no += 1
        new_img_name = "{}-{:05d}.jpg".format(img_prefix, img_no)
        img = plt.imread(img_file)


        img = img[x1:x2, y1:y2]
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img.save(os.path.join(dir, 'input_imgs', new_img_name))
        
        
        
    

    zout = 70
    for img_file in img_files:
        img_no += 1
        new_img_name = "{}-{:05d}.jpg".format(img_prefix, img_no)
        img = plt.imread(img_file)
            

        img = cv2.copyMakeBorder(img, zout, zout, zout, zout, cv2.BORDER_CONSTANT, (0,0,0))
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img.save(os.path.join(dir, 'input_imgs', new_img_name))
        
        


        
    zout = 100
    for img_file in img_files:
        img_no += 1
        new_img_name = "{}-{:05d}.jpg".format(img_prefix, img_no)
        img = plt.imread(img_file)
            

        img = cv2.copyMakeBorder(img, zout, zout, zout, zout, cv2.BORDER_CONSTANT, (0,0,0))
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img.save(os.path.join(dir, 'input_imgs', new_img_name))
        
     
        
        

        
    

        
if __name__ == "__main__":
    dirs = glob.glob(r'dataset\frames_new_2\*')
    dirs = [r'dataset\frames_new_2\duck_position_1']
    for dir in dirs:
        zoom_images(dir)
        
        





    