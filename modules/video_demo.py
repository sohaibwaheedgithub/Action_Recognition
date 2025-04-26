
import cv2
import mediapipe as mp
import numpy as np
from utils import normalize_lmks
from model import LSTM_Model
from openvino.runtime import Core
import constants


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
inference_engine = Core()
model = inference_engine.read_model(r'models\openvino_models\{}\saved_model.xml'.format(constants.MODEL_DIR_NAME))
compiled_model = inference_engine.compile_model(model, device_name='CPU')
output_layer = compiled_model.output(0)


sequence_list = []
action = ''



CLASSES = []
for cl in constants.CLASSES:
  CLASSES.append(cl)
  if not cl in ["left_wave", "right_wave", "no_action_latest"]:
    CLASSES.append(cl + "_back")


  



# For webcam input:
cap = cv2.VideoCapture(r'dataset\test_videos\test_1.mp4')
#cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image = cv2.resize(image, (640, 480), cv2.INTER_AREA)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    pose_landmarks = results.pose_landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    score = ''
    
    if not pose_landmarks == None:

      # Draw the pose annotation on the image.
    
      """mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())"""

      

      frame_height, frame_width = image.shape[0], image.shape[1]
      landmarks = []
      """for lmk in pose_landmarks.landmark:
        landmarks.append(lmk.x * frame_width)
        landmarks.append(lmk.y * frame_height)
        #landmarks.append(lmk.z * frame_width)"""
      for i, lmk in enumerate(pose_landmarks.landmark):
        if i not in [1,3,4,6,9,10,17,18,19,20,21,22,29,30,31,32]:
          landmarks.append(lmk.x * frame_width)
          landmarks.append(lmk.y * frame_height)
          landmarks.append(lmk.z * frame_width)
      landmarks = np.array(landmarks, dtype=np.float32)
      landmarks = np.expand_dims(landmarks, axis = 0)
      normalized_lmks = normalize_lmks(landmarks)
      if not len(sequence_list) == constants.SEQUENCE_LENGTH:
        sequence_list.append(normalized_lmks)
      else:
        sequence_array = np.stack(sequence_list, axis = 1)
        #print(compiled_model([sequence_array])[output_layer][0])
        scores = compiled_model([sequence_array])[output_layer][0]
        idx = np.argmax(scores)
        score = scores[idx]
        action = CLASSES[idx]
        sequence_list.pop(0)
        sequence_list.append(normalized_lmks)
        
    
    #image = cv2.flip(image, 1)
    cv2.putText(
        image, 
        action + ':  ' + str(score),
        (30, 70),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
      )


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()