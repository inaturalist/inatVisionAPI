import random
import time
import datetime

import json
from flask import Flask, request, jsonify, render_template
from forms import ImageForm
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import os
import yaml

config = yaml.safe_load(open("config.yml"))

def _load_taxon_ids(taxa_file):
    taxon_ids = []
    with open(taxa_file) as f:
        for line in f:
            iter, taxon_id = line.rstrip().split(": ")
            taxon_ids.append(int(taxon_id))
    return taxa
TENSORFLOW_TAXON_IDS = _load_taxon_ids("taxa.txt")

app = Flask(__name__)
app.secret_key = config["app_secret"]

UPLOAD_FOLDER = "static/"

graph = None
sess = tf.Session()
with sess.as_default():
    # Load in the graph
    graph_def = tf.GraphDef()
    with open('optimized_model-3.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

sess.graph.finalize()

# Get the input and output operations
input_op = graph.get_operation_by_name('images')
input_tensor = input_op.outputs[0]
output_op = graph.get_operation_by_name('Predictions')
output_tensor = output_op.outputs[0]

@app.route('/', methods=['GET', 'POST'])
def classify():
    form = ImageForm()
    if request.method == 'POST':
        image_file = form.image.data
        extension = os.path.splitext(image_file.filename)[1]
        file_path = os.path.join(UPLOAD_FOLDER, datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')) + extension
        image_file.save(file_path)

        print(file_path)

        # Load in an image to classify and preprocess it
        image = imread(file_path)
        image = imresize(image, [299, 299])
        image = image.astype(np.float32)
        image = (image - 128.) / 128.
        image = image.ravel()
        images = np.expand_dims(image, 0)

        # Get the predictions (output of the softmax) for this image
        preds = sess.run(output_tensor, {input_tensor : images})

        sorted_pred_args = preds[0].argsort()[::-1][:100]
        return jsonify(dict(zip(TENSORFLOW_TAXON_IDS,[ round(elem * 100, 6) for elem in preds[0].astype(float)])))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
