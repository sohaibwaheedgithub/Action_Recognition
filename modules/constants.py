BATCH_SIZE = 32
N_LANDMARKS = 39 

TRAIN_SIZE = 0.75
SEQUENCE_LENGTH = 10
 

MODEL_DIR_NAME = 'SL=10_Frames_Final_No_Hands_No_Jump'

#CLASSES = ['duck', 'jump', 'no_action', 'no_jump', 'right_tilt', 'left_tilt', 'duck_position']

CLASSES = ['duck', 'jump', 'no_action', 'no_action', 'right_tilt', 'left_tilt', 'duck_position']


ACTIONS = {}
for action in CLASSES:
    ACTIONS[action] = {}


ACTIONS['duck']['color'] = (0, 0, 255)
ACTIONS['jump']['color'] = (0, 255, 0)
ACTIONS['left_tilt']['color']= (255, 0, 0)
#ACTIONS['no_jump']['color'] = (160, 160, 160)
ACTIONS['right_tilt']['color'] = (0, 255, 255)
ACTIONS['no_action']['color'] = (160, 160, 160)
ACTIONS['duck_position']['color']= (0, 100, 255)