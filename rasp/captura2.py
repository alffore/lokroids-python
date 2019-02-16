# coding=UTF-8
##
## Código básico para captura de imagenes 1 cada minuto y clasificador CNN
##
## AAFR 29 de marzo de 2018

import time
import socket
from time import sleep
from picamera import PiCamera
import cv2
import tensorflow as tf
import json

dir_base = '/mnt/lokros/imagenes/'
url_base = ''

CATEGORIAS = ['dormido', 'despierto', 'otro']
IMG_SIZE = 70
nombre_modelo = '/home/pi/modeloLokro-multi-4.h5'


def preparaimg(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def guarda_json(datosf):
    """
    :type datosf: Diccionario con datos de la foto
    """
    with open(dir_base + datosf['nimg'] + '.json', 'w') as outfile:
        json.dump(datosf, outfile, ensure_ascii=False)
    return


camera = PiCamera()

camera.vflip = True
camera.hflip = True

camera.resolution = (1024, 768)
camera.start_preview()
# Camera warm-up time
sleep(2)

model = tf.keras.models.load_model(nombre_modelo)

while True:
    datosf = dict()
    id_foto = int(time.time())
    archivo_foto = dir_base + 'lokro' + str(id_foto) + '.jpg'
    camera.capture(archivo_foto, resize=(320, 240))
    datosf['id'] = id_foto
    datosf['imagen_url'] = url_base + 'lokro' + str(id_foto) + '.jpg'
    datosf['nimg'] = 'lokro' + str(id_foto)

    prediction = model.predict([preparaimg(archivo_foto)])
    top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
    datosf['clasificado'] = CATEGORIAS[top_k[0]]

    guarda_json(datosf)
    sleep(60)
