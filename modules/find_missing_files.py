
import glob
import os
import csv
import pandas as pd

img_files = glob.glob(r'C:\Users\sohai\Desktop\Projects\Human_Action_Recognition\dataset\frames_new_2\run_swipe\input_imgs\*')
lmk_file_path  = glob.glob(r'C:\Users\sohai\Desktop\Projects\Human_Action_Recognition\dataset\frames_new_2\run_swipe\input_imgs\*')
lmk_img_files_names = []
inp_img_files_names = []

with open(lmk_file_path, 'r', newline='') as lmk_file:
    csv_reader = csv.reader(lmk_file, delimiter=',')
    for row in csv_reader:
        img_name = row[0]
        lmk_img_files_names.append(img_name)

missing_imgs = []
for img_f in img_files:
    _, img_name = os.path.split(img_f)

    if img_name not in lmk_img_files_names:
        missing_imgs.append(img_name)


print(missing_imgs)



