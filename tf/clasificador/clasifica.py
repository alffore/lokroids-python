##
## Código que clasifica un conjunto de imagenes
##
## AAFR <alffore@gmail.com> 28 de marzo de 2018

import json
import glob
import tensorflow as tf
import sys


def clasifica_image(image_path):
    """Funcion que clasifica imagenes de un directorio"""
    # lee la imagen en image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    aux = dict()
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        aux[human_string] = score

    return aux


def escribe_arch(ah, dato, valor):
    """Escribe el archivo con la clasificación"""
    ah.write(dato + "|" + str(valor['vader']) + "|" + str(valor['c3po']))
    ah.write('\n')


# Comienza el codigo
image_dir_path = sys.argv[1]
print(image_dir_path)

label_lines = [line.rstrip() for line
               in tf.gfile.GFile("retrained_labels.txt")]

with tf.gfile.FastGFile("ratrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

aim = glob.glob(image_dir_path + '/*')

target = open("resultado_clasificado.txt", 'w')

for img in aim:
    print(img)
    res = clasifica_image(img)
    # print(res)
    escribe_arch(target, img, res)

target.close()
