import datetime
import json
import magic
import os
import random
import time
import uuid
import yaml
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from forms import ImageForm
from PIL import Image

config = yaml.safe_load(open("config.yml"))

def _load_taxon_ids(taxa_file):
    taxon_ids = []
    with open(taxa_file) as f:
        for line in f:
            iter, taxon_id = line.rstrip().split(": ")
            taxon_ids.append(int(taxon_id))
    return taxon_ids
TENSORFLOW_TAXON_IDS = _load_taxon_ids("taxa.txt")

app = Flask(__name__)
app.secret_key = config["app_secret"]

UPLOAD_FOLDER = "static/"

tf.config.set_visible_devices([], "GPU")
model = tf.keras.models.load_model("optimized_model2.h5", compile=False)

def write_logstash(image_file, image_uuid, file_path, request_start_datetime, request_start_time, mime_type):
    request_end_time = time.time()
    request_time = round((request_end_time - request_start_time) * 1000, 6)
    logstash_log = open('log/logstash.log', 'a')
    log_data = {'@timestamp': request_start_datetime.isoformat(),
                'uuid': image_uuid,
                'duration': request_time,
                'mime_type': mime_type,
                'client_ip': request.access_route[0],
                'filename': image_file.filename,
                'image_size': os.path.getsize(file_path)}
    json.dump(log_data, logstash_log)
    logstash_log.write("\n")
    logstash_log.close()

@app.route('/', methods=['GET', 'POST'])
def classify():
    form = ImageForm()
    if request.method == 'POST':
        request_start_datetime = datetime.datetime.now()
        request_start_time = time.time()
        image_file = form.image.data
        extension = os.path.splitext(image_file.filename)[1]
        image_uuid = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + extension
        image_file.save(file_path)

        mime_type = magic.from_file(file_path, mime=True)
        # attempt to convert non jpegs
        if mime_type != 'image/jpeg':
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
            file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + '.jpg'
            rgb_im.save(file_path)

        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.central_crop(img, 0.875)
        img = tf.image.resize(img, [299,299])
        img = tf.expand_dims(img,  0)

        preds = model.predict(img_array)
        
        sorted_pred_args = preds[0].argsort()[::-1][:100]
        response_json = jsonify(dict({TENSORFLOW_TAXON_IDS[arg]: round(preds[0][arg] * 100, 6) for arg in sorted_pred_args}))
        write_logstash(image_file, image_uuid, file_path, request_start_datetime, request_start_time, mime_type)
        return response_json
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
