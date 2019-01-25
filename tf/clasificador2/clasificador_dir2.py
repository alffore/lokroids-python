# coding=UTF-8
import cv2
import sys
import os
import tensorflow as tf
import filetype
import json

DATADIR = '/Volumes/COMPARTIDA/devel/Tensorflow/imagenesLokro/imagenes/'  # MacOS casa

CATEGORIAS = ['dormido', 'despierto', 'otro']

IMG_SIZE = int(sys.argv[3])


def preparaimg(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def guarda_json(datosf):
    """
    :type datosf: Diccionario con datos de la foto
    """
    with open(DATADIR + datosf['nimg'] + '.json', 'w') as outfile:
        json.dump(datosf, outfile, ensure_ascii=False)
    return


def marcajson(filepath, tipop):
    with open(filepath) as f:
        data = json.load(f)
        data['clasificado'] = CATEGORIAS[tipop]
        data['comp_clasificado'] = CATEGORIAS[tipop]
        guarda_json(data)


model = tf.keras.models.load_model(sys.argv[1])

if len(sys.argv) > 1:
    path = sys.argv[2]
else:
    path = DATADIR

for img in os.listdir(path):
    tipo_archivo = filetype.guess(os.path.join(path, img))
    if tipo_archivo is not None and tipo_archivo.mime == 'image/jpeg':
        prediction = model.predict([preparaimg(os.path.join(path, img))])
        top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
        print(img + ' ' + str(top_k[0]) + " " + CATEGORIAS[top_k[0]])
        aux_json = img.split('.')
        archivo_json = aux_json[0] + ".json"
        marcajson(os.path.join(path, archivo_json), top_k[0])
