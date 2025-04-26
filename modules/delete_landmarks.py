import os
import shutil
from constants import CLASSES
for action in CLASSES:
    try:
        os.remove(f'dataset/frames_3/{action}/landmarks.csv')
    except FileNotFoundError as e:
        print(e)
        continue
    
    """try:
        shutil.rmtree(f'dataset/frames/{action}/output_imgs')
    except FileNotFoundError as e:
        print(e)
        continue"""