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

def _load_taxa_labels(taxa_file):
    taxa = {}
    with open(taxa_file) as f:
        for line in f:
            iter, taxon = line.rstrip().split(": ")
            taxa[int(iter)] = int(taxon)
    return taxa
taxa = _load_taxa_labels("taxa.txt")

app = Flask(__name__)
app.config.from_object(__name__)
app.debug = True
app.secret_key = "asdlkfjlkdsjklds"

UPLOAD_FOLDER = 'static/'

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
        t = time.time()
        preds = sess.run(output_tensor, {input_tensor : images})
        dt = time.time() - t

        print("Execution time: %0.2f" % (dt * 1000.))

        # The probabilities should sum to 1
        assert np.isclose(np.sum(preds[0]), 1)

        sorted_pred_args = preds[0].argsort()[::-1][:10]
        top10_preds = {taxa[arg]: preds[0][arg] for arg in sorted_pred_args}
        return render_template(
            'results.html',
            name='alex',
            preds=top10_preds,
            image_path = file_path
        )
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)
