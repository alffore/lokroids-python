##
## Código básico para captura de imagenes 1 cada minuto
##
## AAFR 29 de marzo de 2018

from time import sleep
from picamera import PiCamera

dir_base = '/mnt/lokros/imagenes/'

camera = PiCamera()

camera.vflip = True
camera.hflip = True

camera.resolution = (1024, 768)
camera.start_preview()
# Camera warm-up time
sleep(2)

id_foto = 1
while True:
    camera.capture(dir_base + 'lokro' + str(id_foto) + '.jpg', resize=(320, 240))
    id_foto += 1
    if id_foto > 700:
        id_foto = 1
    sleep(60)
