import cv2
import constants
import numpy as np
import mediapipe as mp
from openvino.runtime import Core
from utils import normalize_lmks, draw_bounding_box



class Realtime_Demo():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
            )
        

        model = Core().read_model(r'models\openvino_models\{}\saved_model.xml'.format(constants.MODEL_DIR_NAME))
        self.compiled_model = Core().compile_model(model, device_name='CPU')
        del model
        self.output_layer = self.compiled_model.output(0)

        self.sequence_list = []
        
        self.current_action = 'no_action'

        self.actions = constants.ACTIONS

        self.score = 0


        self.left_tilts = 0
        self.right_tilts = 0
        

        self.shoulder_dists = []

        self.left_ankle_ys = []
        self.right_ankle_ys = []

        self.left_shoulder_ys = []
        self.right_shoulder_ys = []


        self.freeze = False
        self.freeze_counts = 0

        self.actions_list = []

        self.prev_action = ''



            


    def run_demo(self, cam_key, approach_type, video_path = ''):
        cap = cv2.VideoCapture(cam_key)
        
        
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        writer = cv2.VideoWriter(
           video_path,
           cv2.VideoWriter_fourcc(*'mp4v'),
           fps,
           (width, height)
        )


        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break             


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            pose_landmarks = results.pose_landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        

            if not pose_landmarks == None:
                # Draw the pose annotation on the image.
                '''mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())'''

                frame_height, frame_width = image.shape[0], image.shape[1]
                landmarks = []
                for i, lmk in enumerate(pose_landmarks.landmark):
                    
                    # To get the required landmarks
                    if i == 11:
                        left_shoulder_x = lmk.x * frame_width
                        left_shoulder_y = lmk.y * frame_height
                        
                        if not len(self.left_shoulder_ys) == constants.SEQUENCE_LENGTH:
                            self.left_shoulder_ys.append(left_shoulder_y)
                        else:
                            self.left_shoulder_ys.pop(0)
                            self.left_shoulder_ys.append(left_shoulder_y)


                    elif i == 12:
                        right_shoulder_x = lmk.x * frame_width
                        right_shoulder_y = lmk.y * frame_height

                        shoulder_dist = np.sqrt(((right_shoulder_x - left_shoulder_x)**2) + ((right_shoulder_y - left_shoulder_y)**2))

                        if not len(self.shoulder_dists) == constants.SEQUENCE_LENGTH:
                            self.shoulder_dists.append(shoulder_dist)
                        else:
                            self.shoulder_dists.pop(0)
                            self.shoulder_dists.append(shoulder_dist)


                        if not len(self.right_shoulder_ys) == constants.SEQUENCE_LENGTH:
                            self.right_shoulder_ys.append(right_shoulder_y)
                        else:
                            self.right_shoulder_ys.pop(0)
                            self.right_shoulder_ys.append(right_shoulder_y)

                    
                    elif i == 23:
                        left_hipjoint_x = lmk.x * frame_width
                        left_hipjoint_y = lmk.y * frame_height
                    elif i == 24:
                        right_hipjoint_x = lmk.x * frame_width
                        right_hipjoint_y = lmk.y * frame_height


                    elif i == 27:
                        left_ankle_x = lmk.x * frame_width
                        left_ankle_y = lmk.y * frame_height

                        
                        left_dist = np.sqrt(((left_ankle_x - left_hipjoint_x)**2) + ((left_ankle_y - left_hipjoint_y)**2))
                        left_ratio = shoulder_dist / left_dist


                        if not len(self.left_ankle_ys) == constants.SEQUENCE_LENGTH:
                            self.left_ankle_ys.append(left_ankle_y)
                        else:
                            self.left_ankle_ys.pop(0)
                            self.left_ankle_ys.append(left_ankle_y)

                        
                    elif i == 28:
                        right_ankle_x = lmk.x * frame_width
                        right_ankle_y = lmk.y * frame_height


                        if not len(self.right_ankle_ys) == constants.SEQUENCE_LENGTH:
                            self.right_ankle_ys.append(right_ankle_y)
                        else:
                            self.right_ankle_ys.pop(0)
                            self.right_ankle_ys.append(right_ankle_y)



                        right_dist = np.sqrt(((right_ankle_x - right_hipjoint_x)**2) + ((right_ankle_y - right_hipjoint_y)**2))

                        right_ratio = shoulder_dist / right_dist


                    elif i == 31:
                        left_foot_x = lmk.x * frame_width
                        left_foot_y = lmk.y * frame_height


                    if i in [0, 2, 5, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28]:
                    
                        landmarks.append(lmk.x * frame_width)
                        landmarks.append(lmk.y * frame_height)
                        landmarks.append(lmk.z * frame_width)

            


                landmarks = np.array(landmarks, dtype=np.float32)
                landmarks = np.expand_dims(landmarks, axis = 0)
                normalized_lmks = normalize_lmks(landmarks)
                
                if not self.freeze:

                    if not len(self.sequence_list) == constants.SEQUENCE_LENGTH:
                        self.sequence_list.append(normalized_lmks)
                        
                    else:
                        sequence_array = np.stack(self.sequence_list, axis = 1)
                        scores = self.compiled_model([sequence_array])[self.output_layer][0]
                        idx = np.argmax(scores)
                        self.current_action = constants.CLASSES[idx]

                        self.score = scores[idx]


                        # Logic For detecting tilts
                        diff = right_shoulder_y - left_shoulder_y
                        if diff > 20 and not self.current_action == 'right_kick':       # Adult Mode threshold is 30
                            self.current_action = 'right_tilt'
                        
                        elif diff < -20 and not self.current_action == 'left_kick':
                            self.current_action = 'left_tilt'
                            
                        

                        if not approach_type == 'Artificial Intelligence':

                            # Logic for detecting jump using lmks                    
                            right_shoulder_y_diff = self.right_shoulder_ys[0] - self.right_shoulder_ys[-1]
                            left_shoulder_y_diff = self.left_shoulder_ys[0] - self.left_shoulder_ys[-1]

                            right_ankle_y_diff = self.right_ankle_ys[0] - self.right_ankle_ys[-1]
                            left_ankle_y_diff = self.left_ankle_ys[0] - self.left_ankle_ys[-1]


                            if (right_shoulder_y_diff > 10) and (left_shoulder_y_diff > 10) and (right_ankle_y_diff > 10) and (left_ankle_y_diff > 10):
                                self.current_action = 'jump'
                    

                        # Logic for detecting duck position using lmks
                        
                        if right_ratio >= 0.90 and left_ratio >= 0.90:      # Adult Mode threshold is 0.55
                            self.current_action = 'duck'


                                                
                        # Logic for detecting forward-backward movement
                        shoulder_diff = self.shoulder_dists[-1] - self.shoulder_dists[0]
                        if (shoulder_diff > 10) or (shoulder_diff < -10):
                            self.current_action = 'no_action'


                        if self.prev_action == 'duck' and (self.current_action == 'jump' or self.current_action == 'no_action'):
                            self.freeze = True
                            self.current_action = 'no_action'

                        

                        



                else:
                    self.freeze_counts += 1
                    self.current_action = 'no_action'
                    if self.freeze_counts == 20:
                        self.freeze = False
                        self.freeze_counts = 0


                

                # To draw bounding box
                image = draw_bounding_box(
                    image, 
                    self.current_action, 
                    right_shoulder_x,
                    right_shoulder_y,
                    left_foot_x,
                    left_foot_y
                )
                

                self.sequence_list.pop(0)
                self.sequence_list.append(normalized_lmks)


                self.prev_action = self.current_action



                    
            
            if video_path != '':
                writer.write(image)

            
            image = cv2.resize(image, (1200, 780))

        
            
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_demo = Realtime_Demo()
    cam_key = r'test_videos\video1.mp4'
    cam_key = 0
    approach_type = 'Artificial Intelligence'
    #approach_type = 'Programming Logics'
    realtime_demo.run_demo(cam_key, approach_type, 'saved_videos\demo_video2.mp4')