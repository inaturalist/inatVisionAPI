import time
import os
import urllib
import uuid
import json

from flask import Flask, request, render_template
from web_forms import ImageForm
from inat_inferrer import InatInferrer
from model_results import ModelResults


class InatVisionAPI:

    def __init__(self, config):
        self.setup_inferrer(config)
        self.app = Flask(__name__)
        self.app.secret_key = config["app_secret"]
        self.upload_folder = "static/"
        self.app.add_url_rule(
            "/", "index", self.index_route, methods=["GET", "POST"])

    def setup_inferrer(self, config):
        self.inferrer = InatInferrer(config)

    def top_x_results(self, results, x):
        top_x = dict(sorted(
            results.scores["combined"].items(), key=lambda x: x[1], reverse=True)[:x])
        to_return = []
        for index, arg in enumerate(top_x):
            to_return.append({
                "combined_score": round(results.scores["combined"][arg] * 100, 6),
                "vision_score": round(results.scores["vision"][arg] * 100, 6),
                "geo_score": round(results.scores["geo"][arg] * 100, 6),
                "id": self.inferrer.taxonomy.taxa[arg].id,
                "name": self.inferrer.taxonomy.taxa[arg].name,
                "index": index
            })
        return to_return

    def index_route(self):
        form = ImageForm()
        if "observation_id" in request.args:
            observation_id = request.args["observation_id"]
        else:
            observation_id = form.observation_id.data
        if request.method == "POST" or observation_id:
            lat = form.lat.data
            lng = form.lng.data
            file_path = None
            image_uuid = None
            if observation_id:
                image_uuid = "downloaded-obs-" + observation_id
                file_path, lat, lng, iconic_taxon_id = self.download_observation(
                    observation_id, image_uuid)
            else:
                image_uuid = str(uuid.uuid4())
                file_path = self.process_upload(form.image.data, image_uuid)
                iconic_taxon_id = None
            if file_path is None:
                return render_template("home.html")

            image = self.inferrer.prepare_image_for_inference(file_path, image_uuid)

            # Vision
            vision_start_time = time.time()
            vision_results = self.inferrer.vision_inferrer.process_image(image, iconic_taxon_id)
            vision_total_time = time.time() - vision_start_time
            print("Vision Time: %0.2fms" % (vision_total_time * 1000.))

            # Geo
            geo_start_time = time.time()
            if lat is not None and lat != "" and lng is not None and lng != "":
                geo_results = self.inferrer.geo_model.predict(lat, lng, iconic_taxon_id)
            else:
                geo_results = {}
            geo_total_time = time.time() - geo_start_time
            print("GeoTime: %0.2fms" % (geo_total_time * 1000.))

            # Scoring
            scoring_start_time = time.time()
            results = ModelResults(vision_results, geo_results, self.inferrer.taxonomy)
            results.aggregate_scores()
            scoring_total_time = time.time() - scoring_start_time
            print("Score Time: %0.2fms" % (scoring_total_time * 1000.))

            results.print()
            return {
                "common_ancestor": None if results.fine_common_ancestor is None else {
                    "id": results.fine_common_ancestor.id,
                    "name": results.fine_common_ancestor.name
                },
                "rough_common_ancestor": None if results.common_ancestor is None else {
                    "id": results.common_ancestor.id,
                    "name": results.common_ancestor.name
                },
                "top_combined": self.top_x_results(results, 20)
            }

        else:
            return render_template("home.html")

    def process_upload(self, form_image_data, image_uuid):
        if form_image_data is None:
            return None
        extension = os.path.splitext(form_image_data.filename)[1]
        file_path = os.path.join(self.upload_folder, image_uuid) + extension
        form_image_data.save(file_path)
        return file_path

    # Fetch the observation metadata and first image using the iNaturalist API
    def download_observation(self, observation_id, image_uuid):
        url = "https://api.inaturalist.org/v1/observations/" + observation_id
        cache_path = os.path.join(self.upload_folder, image_uuid) + ".jpg"
        # fetch observation details using the API
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        if (data is None or data["results"] is None or data["results"][0] is
            None or data["results"][0]["photos"] is None or data["results"][0]["photos"][0] is
                None or data["results"][0]["photos"][0]["url"] is None):
            return None, None, None
        # download the first image if it isn't already cached
        if not os.path.exists(cache_path):
            urllib.request.urlretrieve(
                data["results"][0]["photos"][0]["url"].replace("square", "medium"), cache_path)
        latlng = data["results"][0]["location"].split(",")
        # return the path to the cached image, coordinates, and iconic taxon
        return cache_path, latlng[0], latlng[1], data["results"][0]["taxon"]["iconic_taxon_id"]
