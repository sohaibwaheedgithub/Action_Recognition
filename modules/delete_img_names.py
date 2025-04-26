import glob 
import os
import pandas as pd

lmk_file_path  = r'dataset\frames_final\duck_position\landmarks.csv'
df = pd.read_csv(lmk_file_path, header = None)
df.drop([0], axis = 1, inplace=True)

df.to_csv(r'dataset\frames_final\duck_position\landmarks_updated.csv', header=None, index=False)