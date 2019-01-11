# coding=UTF-8
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import os
import cv2
from tqdm import tqdm
import random
import filetype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
import time

# DATADIR = "/media/alfonso/COMPARTIDA/devel/Tensorflow/imagenesLokro/imagenes_clas/"

DATADIR = '/Volumes/COMPARTIDA/devel/Tensorflow/imagenesLokro/imagenes_clas/'

# CATEGORIAS = ['dormido', 'despierto', 'otro', 'barriba', 'ausente']
CATEGORIAS = ['dormido', 'despierto', 'otro']

IMG_SIZE = 70

NAME = "bz16-adam-c5-{}".format(int(time.time()))

training_data = []

def create_training_data():
    for category in CATEGORIAS:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIAS.index(category)  # get the classification  (0 or a 1). 0=dormido 1=despierto

        for img in tqdm(os.listdir(path)):
            tipo_archivo = filetype.guess(os.path.join(path, img))
            if tipo_archivo is not None and tipo_archivo.mime == 'image/jpeg':
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:
                    pass


create_training_data()

print("Tamaño del conjunto de entrenamiento: " + str(len(training_data)))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X / 255.0

model = Sequential()

model.add(Flatten())

model.add(Dense(648))
model.add(Activation('relu'))

model.add(Dense(648))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.fit(X, np.array(y), batch_size=8,
          epochs=20,
          validation_split=0.3,
          callbacks=[tensorboard])

# model.save("modeloLokro3m.h5")
