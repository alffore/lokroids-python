# coding=UTF-8
import cv2
import sys
import os
import tensorflow as tf
import filetype


CATEGORIAS = ['dormido', 'despierto', 'otro']


def preparaimg(filepath):
    # IMG_SIZE = 240
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model(sys.argv[1])

path = sys.argv[2]

for img in os.listdir(path):
    tipo_archivo = filetype.guess(os.path.join(path, img))
    if tipo_archivo is not None and tipo_archivo.mime == 'image/jpeg':
        prediction = model.predict([preparaimg(os.path.join(path, img))])
        print(img + " " + str(prediction[0]))
        # print(img + " " + str(np.dot(prediction[0], [1, 0, 0])))
        top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
        print(img+' '+str(top_k[0])+" "+CATEGORIAS[top_k[0]])
