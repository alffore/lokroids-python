import picamera
from time import sleep

camera = picamera.PiCamera()

camera.vflip = True
camera.hflip = True

camera.capture('image1.jpg')

sleep(5)

camera.capture('image2.jpg')

