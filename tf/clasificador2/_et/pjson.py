import sys
import os
import json

from pprint import pprint

DATADIR = '/Volumes/COMPARTIDA/devel/Tensorflow/imagenesLokro/imagenes/'  # MacOS casa

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = DATADIR

for img in os.listdir(path):

    aux = img.split('.')
    if aux[1] == 'json':
        print(os.path.join(path, img))
        with open(os.path.join(path, img)) as f:
            data = json.load(f)
            pprint(data)
            print(data["nimg"])
