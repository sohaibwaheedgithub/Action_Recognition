import os
import constants
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTMCell, Dense, Input, RNN, StackedRNNCells
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from utils import normalize_lmks



class Preprocessing_Layer1(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def call(self, _X):
        _X = tf.transpose(_X, [1,0,2])
        _X = tf.reshape(_X, [-1, constants.N_LANDMARKS])
        return _X

    

class Preprocessing_Layer2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def call(self, _X):
        
        _X = tf.split(_X, constants.SEQUENCE_LENGTH, 0)
        _X = tf.stack(_X, axis = 0)
        return _X
        



class LSTM_Model():
    def __init__(self):
        self.model = None
        self.n_classes = len(constants.CLASSES)




    def build_model(self):

        input1 = Input(shape = [None, constants.N_LANDMARKS])
        pre_layer1 = Preprocessing_Layer1()(input1)
        dense_layer_1 = Dense(34, activation = 'relu')(pre_layer1)
        pre_layer2 = Preprocessing_Layer2()(dense_layer_1)

        
        lstm_cell_1 = LSTMCell(34)
        lstm_cell_2 = LSTMCell(34)
        lstm_cells = StackedRNNCells([lstm_cell_1, lstm_cell_2])
        lstm_layer = RNN(lstm_cells, unroll = True, time_major = True)(pre_layer2)
    
        dense_layer_2 = Dense(self.n_classes, activation = 'softmax')(lstm_layer)

        self.model = Model(inputs = [input1], outputs = [dense_layer_2])


        self.model.compile(
            loss = 'sparse_categorical_crossentropy',
            optimizer = Adam(),
            metrics = 'accuracy'
        )


    def train_model(self, X_train, y_train, X_valid, y_valid, model_path):
        earlystopping_cb = EarlyStopping(patience=10)
        modelcheckpoint_cb = ModelCheckpoint(model_path)
        reducelronplateau_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-10)
        
        
        self.model.fit(
            X_train,
            y_train,
            batch_size = constants.BATCH_SIZE,
            epochs = 200,
            validation_data = (X_valid, y_valid),
            callbacks = [modelcheckpoint_cb, earlystopping_cb, reducelronplateau_cb]
        )
        

    
    def load_model(self, model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects = {
                'Preprocessing_Layer1': Preprocessing_Layer1,
                'Preprocessing_Layer2': Preprocessing_Layer2
                }
            )
        return model

    
    def save_model(self, model, model_dir):
        tf.saved_model.save(model, model_dir)



if __name__ == "__main__":
    
    import csv
    import glob
    import numpy as np
    import tensorflow_datasets as tfds

    
    
    def load_data():
        lmk_files = glob.glob(r'dataset\frames_new_3\*\landmarks.csv')
        #assert len(lmk_files) == len(constants.CLASSES), 'lmk_files are not equal to the number of classes'
        #print(constants.CLASSES)
        
        
        final_X = []
        final_y = []
        array_length = 0
        class_id = 0
        for file in lmk_files:
            root, _ = os.path.split(file)
            _, action_name = os.path.split(root)
            print("In " + action_name)
            with open(file, 'r', newline="") as f:
                csv_reader = csv.reader(f, delimiter=",")
                list_of_Xs = []
                for raw in csv_reader:
                    X = np.array(raw)
                    list_of_Xs.append(X)

                
                X = np.array(list_of_Xs, dtype=np.float32)
                X = normalize_lmks(X)

                X = tf.data.Dataset.from_tensor_slices(X)
                X = X.batch(constants.SEQUENCE_LENGTH, drop_remainder=True)
                X = tfds.as_numpy(X)
                X = np.array(list(X), dtype = np.float32)
                array_length += X.shape[0]
                y = np.array([class_id] * X.shape[0], dtype = np.uint8)
                y = np.expand_dims(y, axis = -1)
                class_id += 1

                final_X.append(X)
                final_y.append(y)

                print(X.shape, y.shape)

          

        final_X = np.concatenate(final_X, axis = 0)
        final_y = np.concatenate(final_y, axis = 0)
        assert final_X.shape[0] == array_length, "Length of final_X is not equal to array_length"
        assert final_y.shape[0] == array_length, "Length of final_y is not equal to array_length"
        
        
        return final_X, final_y


    
    def load_data_10():
        lmk_files = glob.glob(r'dataset\frames_final\*\landmarks.csv')
        #assert len(lmk_files) == len(constants.CLASSES), 'lmk_files are not equal to the number of classes'
        #print(constants.CLASSES)
        
        final_X = []
        final_y = []
        array_length = 0
        class_id = 0
        for file in lmk_files:
            root, _ = os.path.split(file)
            _, action_name = os.path.split(root)
            print("In " + action_name)
            


            if action_name in ['jump', 'duck', 'duck_back', 'no_jump']:
                
                valid_index = 0
                
                if action_name in ['duck_back', 'no_jump']:
                    valid_index += 6

                with open(file, 'r', newline="") as f:
                    csv_reader = csv.reader(f, delimiter=",")
                    list_of_Xs = []
                    seq_counter = 0
                    for idx, raw in enumerate(csv_reader):
                        if idx == valid_index:
                            X = np.array(raw)
                            list_of_Xs.append(X)
                            valid_index += 1
                            seq_counter += 1
                            if seq_counter == constants.SEQUENCE_LENGTH:
                                valid_index += 6
                                seq_counter = 0
            else:
                print("For {}, i am in else block".format(action_name))
                with open(file, 'r', newline="") as f:
                    csv_reader = csv.reader(f, delimiter=",")
                    list_of_Xs = []
                    for raw in csv_reader:
                        X = np.array(raw)
                        list_of_Xs.append(X)
                        


                
            X = np.array(list_of_Xs, dtype=np.float32)
            
            X = X[:, list(range(0, 21)) + list(range(33, 51))]
            

            X = normalize_lmks(X)

            X = tf.data.Dataset.from_tensor_slices(X)
            X = X.batch(constants.SEQUENCE_LENGTH, drop_remainder=True)
            X = tfds.as_numpy(X)
            X = np.array(list(X), dtype = np.float32)
            array_length += X.shape[0]
            y = np.array([class_id] * X.shape[0], dtype = np.uint8)
            y = np.expand_dims(y, axis = -1)
            class_id += 1

            final_X.append(X)
            final_y.append(y)

            print(X.shape, y.shape)


          

        final_X = np.concatenate(final_X, axis = 0)
        final_y = np.concatenate(final_y, axis = 0)

        
        assert final_X.shape[0] == array_length, "Length of final_X is not equal to array_length"
        assert final_y.shape[0] == array_length, "Length of final_y is not equal to array_length"
        
        
        return final_X, final_y

    
    def setup():
        X, y = load_data_10()
        print(np.unique(y, return_counts=True))
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            train_size = constants.TRAIN_SIZE,
            random_state=42,
            shuffle=True
        )

        return X_train, y_train, X_valid, y_valid
    

    
    lstm_model = LSTM_Model()
    
    
    
    keras_model_dir = r'models\keras_models\{}'.format(constants.MODEL_DIR_NAME)
    saved_model_dir = r'models\saved_models\{}'.format(constants.MODEL_DIR_NAME)
    model_path = r'models\keras_models\{}\model.h5'.format(constants.MODEL_DIR_NAME)
    
    
    os.mkdir(keras_model_dir)
    os.mkdir(saved_model_dir)
    
    
    
    X_train, y_train, X_valid, y_valid = setup()
    
    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)


    
    lstm_model.build_model()
    lstm_model.train_model(X_train, y_train, X_valid, y_valid, model_path)
    
    
    
    model = lstm_model.load_model(model_path)
    tf.saved_model.save(model, saved_model_dir)