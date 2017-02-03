import os, classifier, datetime
from flask import Flask, render_template, request, jsonify
from forms import ImageForm
from PIL import Image
import json
import yaml

config = yaml.safe_load(open("config.yml"))

CAFFE_MODEL = config["model_file"]
DEPLOY_FILE = config["deploy_file"]
MEAN_FILE = config["mean_file"]
LABELS_FILE = config["labels_file"]
UPLOAD_FOLDER = config["upload_folder"]

# this is model-specific
def pre_process(filepath):
    size=(224,224)
    im = Image.open(filepath)
    return im.resize(size)

app = Flask(__name__)
app.debug = True
app.secret_key = config["app_secret"]

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageForm()
    if request.method == 'POST':
        image_file = form.image.data
        extension = os.path.splitext(image_file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')) + extension
        image_file.save(filepath)
        pre_process(filepath).save(filepath)

        image_files = [filepath]
        classifications = classifier.classify(
            caffemodel=CAFFE_MODEL,
            deploy_file=DEPLOY_FILE,
            image_files=image_files,
            labels_file=LABELS_FILE,
            mean_file=MEAN_FILE,
            use_gpu=True
        )

        return jsonify(classifications)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
