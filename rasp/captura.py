##
## Codigo basico para captura de imagenes 1 cada minuto
##
## AAFR 29 de marzo de 2018

import time
import json
from time import sleep
from picamera import PiCamera

dir_base = '/mnt/lokros/imagenes/'
url_base = '//127.0.0.1:3000/imagenes/'


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

while True:
    datosf = dict()
    id_foto = int(time.time())
    camera.capture(dir_base + 'lokro' + str(id_foto) + '.jpg', resize=(320, 240))
    datosf['id'] = id_foto
    datosf['imagen_url'] = url_base + 'lokro' + str(id_foto) + '.jpg'
    datosf['nimg'] = 'lokro' + str(id_foto)
    datosf['clasificado'] = ''
    guarda_json(datosf)
    sleep(60)
