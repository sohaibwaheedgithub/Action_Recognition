# Importing Libraries

import cv2
import numpy as np
from constants import ACTIONS


def normalize_lmks(lmks):
    lmks_mins = np.expand_dims(np.min(lmks, axis=-1), 1)
    lmks_maxs = np.expand_dims(np.max(lmks, axis=-1), 1)
    lmks_ranges = lmks_maxs - lmks_mins

    normalized_lmks = (lmks - lmks_mins) / lmks_ranges

    return normalized_lmks



def draw_bounding_box(image, current_action, x1, y1, x2, y2):
    
    cv2.rectangle(
        image,
        (int(x1 - 20), int(y1 - 60)),
        (int(x2 + 23), int(y2)),
        ACTIONS[current_action]['color'],
        3
    )

    cv2.rectangle(
        image,
        (int(x1 - 20), int(y1 - 90)),
        (int(x2 + 23), int(y1 - 60)),
        ACTIONS[current_action]['color'],
        -1
    )

    cv2.putText(
        image,
        current_action,
        (int(x1 - 20), int(y1 - 60)),
        0,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image
