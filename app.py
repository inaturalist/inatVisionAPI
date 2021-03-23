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

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

from wtforms import IntegerField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

import cv2
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
model = tf.keras.models.load_model("optimized_model.h5", compile=False)

class CamForm(FlaskForm):
    target_taxon = IntegerField("target_taxon", validators=[DataRequired()])
    image = FileField(validators=[FileRequired()])

# using https://www.kaggle.com/gowrishankarin/gradcam-model-interpretability-vgg16-xception
# as a guide
# given an iNat xception classiifier model, make a cam model
def _make_cam_split_models(classifier_model, last_conv_layer_name, classifier_layer_names):
    # split the xception classifier model into two parts, one 
    # from input to final conv layer, another from the final
    # conv layer to the class predictions

    # make a model from input to final conv layer
    last_conv_layer = classifier_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.Model(classifier_model.inputs, last_conv_layer.output)

    # make another model, from final conv layer to class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    post_conv_classifier_model = tf.keras.Model(classifier_input, x)

    return (conv_model, post_conv_classifier_model)

(cam_conv_model, cam_classifier) = _make_cam_split_models(
    model,
    "block14_sepconv2_act",     # xception final conv layer
    ["global_average_pooling2d", "dropout", "dense_logits", "predictions"]  # post-conv layers in our xception model
)



def _make_heatmap(conv_model, post_conv_classifier_model, image, target_class_idx):

    with tf.GradientTape() as tape:
        # compute the activations of the last conv layer and make the tape watch it
        conv_model_output = conv_model(image)
        tape.watch(conv_model_output)

        # get the predictions and the channel for the target class index
        preds = post_conv_classifier_model(conv_model_output)
        class_channel = preds[:, target_class_idx]

        # using tape, get the gradient for the predicted class wrt to the output
        # feature map of the conv model output
        grads = tape.gradient(class_channel, conv_model_output)

        # calculate the mean intensity of the gradient over its feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_model_output = conv_model_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()

        # multiply each channel in the feature map array by weight importance of the channel
        for i in  range(pooled_grads.shape[-1]):
            conv_model_output[:, :, i] *= pooled_grads[i]
        
        # the channel-wise mean of the resulting feature map is our heatmap of class activation
        heatmap = np.mean(conv_model_output, axis=-1)

        # normalize the meatmap between [0, 1]
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

def _superimpose_heatmap(image_path, heatmap):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))

    heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
    superimposed_img = np.uint8(superimposed_img)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img



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

        preds = model.predict(img)
        
        sorted_pred_args = preds[0].argsort()[::-1][:100]
        print(sorted_pred_args)
        response_json = jsonify(dict({TENSORFLOW_TAXON_IDS[arg]: round(preds[0][arg] * 100, 6) for arg in sorted_pred_args}))
        write_logstash(image_file, image_uuid, file_path, request_start_datetime, request_start_time, mime_type)
        return response_json
    else:
        return render_template('home.html')

@app.route('/grad-cam', methods=['GET', 'POST'])
def grad_cam():
    form = CamForm()
    if request.method == 'POST':
        target_taxon_id = form.target_taxon.data

        if not target_taxon_id in TENSORFLOW_TAXON_IDS:
            # we can't made a heatmap for a target taxon that's not in the model
            # should render an alert or soemthing?
            return 
        
        # find the target class/leaf id (the id that's in the model)
        target_leaf_id = TENSORFLOW_TAXON_IDS.index(target_taxon_id)

        # extract the image from the POST and save it
        image_file = form.image.data
        filename = secure_filename(image_file.filename)
        image_uuid = str(uuid.uuid4())
        file_path =  os.path.join(UPLOAD_FOLDER, image_uuid) + ".jpg"
        image_file.save(file_path)

        mime_type = magic.from_file(file_path, mime=True)
        # attempt to convert non jpegs
        if mime_type != 'image/jpeg':
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
            file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + '.jpg'
            rgb_im.save(file_path)

        # read the image in as a normalized array
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        orig_size = img.shape
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [299,299])
        img = tf.expand_dims(img, 0)

        # make the heatmap
        heatmap = _make_heatmap(cam_conv_model, cam_classifier, img, target_leaf_id)
        heatmap_uuid = str(uuid.uuid4())
        heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_uuid) + ".jpg"
        
        # convert the heatmap into the desired colormap
        # in our case, cv2.COLORMAP_JET 
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        heatmap_img = tf.keras.preprocessing.image.array_to_img(heatmap_img)
        heatmap_img = heatmap_img.resize((orig_size[1], orig_size[0]))
        heatmap_img.save(heatmap_path)

        # superimpose the heatmap over the original image
        superimposed = _superimpose_heatmap(file_path, heatmap)
        cam_uuid = str(uuid.uuid4())
        cam_path = os.path.join(UPLOAD_FOLDER, cam_uuid) + ".jpg"
        superimposed.save(cam_path)
        
        return render_template("results.html", superimposed_img = cam_path, heatmap_img = heatmap_path)
    else:
        return render_template('cam.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6009)
