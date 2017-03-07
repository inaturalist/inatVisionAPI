import os, datetime
from client.classifier import InatClassifier
from flask import Flask, render_template, request, jsonify
from forms import ImageForm
from werkzeug.utils import secure_filename
import json
import yaml

config = yaml.safe_load(open("config.yml"))

TAXA_FILE = config['taxa_file']
TFSERVING_HOST = config['tfserving_host']
TFSERVING_PORT = config['tfserving_port']

classifier = InatClassifier(
    TAXA_FILE,
    TFSERVING_HOST,
    TFSERVING_PORT
)

app = Flask(__name__)
app.debug = True
app.secret_key = config["app_secret"]

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageForm()
    if request.method == 'POST':
        image_data = form.image.data.read()
        return jsonify(classifier.classify_data(image_data))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
