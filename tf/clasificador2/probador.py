# coding=UTF-8
import cv2
import sys
import tensorflow as tf

CATEGORIAS = ['dormido', 'despierto', 'otro']


def preparaimg(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model(sys.argv[1])
prediction = model.predict([preparaimg(sys.argv[2])])

print('Modelo: '+sys.argv[1])
print(sys.argv[2]+' Clasificado: '+CATEGORIAS[int(prediction[0][0])]+' ('+str(prediction[0][0])+')')
print(prediction)

