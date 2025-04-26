import glob 
import os
import pandas as pd

lmk_file_path  = r'C:\Users\sohai\Desktop\Projects\Human_Action_Recognition\dataset\frames_4_1\left_punch\landmarks.csv'
df = pd.read_csv(lmk_file_path, header = None)
df = df[:((df.shape[0] // 5) * 3)]


df.to_csv(r'C:\Users\sohai\Desktop\Projects\Human_Action_Recognition\dataset\frames_4_1\left_punch\landmarks_shrinked.csv', header=None, index=False)