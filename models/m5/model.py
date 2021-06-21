from models.FaceAntiSpoofing import FaceAntiSpoofingInterface
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.models
SIZE = 64
from keras import backend as K


class M5FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.model = self.get_model()

    def get_model(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (3, SIZE, SIZE)
        else:
            input_shape = (SIZE, SIZE, 3)

        model = keras.models.Sequential()
        model.add(Conv2D(32, (2, 2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.load_weights("models/m5/files/weights_liveness.h5")
        return model

    def get_real_score(self, bgr, face_bbox):
        crop = bgr[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2], :]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        cut = cv2.resize(crop_rgb, (64, 64)) / 255.
        with self.graph.as_default():
            with self.session.as_default():
                real_score = self.model.predict(np.asarray([cut]))[0]

        return real_score

