import datetime
import time
import os
import urllib
import uuid
import json

from flask import Flask, request, render_template
from web_forms import ImageForm
from inat_inferrer import InatInferrer
from model_scoring import ModelScoring


class InatVisionAPI:

    def __init__(self, config):
        self.setup_inferrer(config["models"][0])
        self.app = Flask(__name__)
        self.app.secret_key = config["app_secret"]
        self.upload_folder = "static/"
        self.app.add_url_rule(
            "/", "index", self.index_route, methods=["GET", "POST"])

    def setup_inferrer(self, config):
        self.inferrer = InatInferrer(config)

    def index_route(self):
        form = ImageForm()
        if "observation_id" in request.args:
            observation_id = request.args["observation_id"]
        else:
            observation_id = form.observation_id.data
        if "geomodel" in request.args:
            geomodel = request.args["geomodel"]
        else:
            geomodel = form.geomodel.data
        if request.method == "POST" or observation_id:
            request_start_datetime = datetime.datetime.now()
            request_start_time = time.time()
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
            scores = self.score_image(image, lat, lng, iconic_taxon_id, geomodel)
            self.write_logstash(image_uuid, file_path, request_start_datetime, request_start_time)
            return scores
        else:
            return render_template("home.html")

    def score_image(self, image, lat, lng, iconic_taxon_id, geomodel):
        # Vision
        vision_start_time = time.time()
        vision_scores = self.inferrer.vision_inferrer.process_image(image, iconic_taxon_id)
        vision_total_time = time.time() - vision_start_time
        print("Vision Time: %0.2fms" % (vision_total_time * 1000.))

        if geomodel != "true":
            top_x = dict(sorted(
                vision_scores.items(), key=lambda x: x[1], reverse=True)[:100])
            to_return = {}
            for index, arg in enumerate(top_x):
                to_return[arg] = round(vision_scores[arg] * 100, 6)
            return to_return

        # Geo
        geo_start_time = time.time()
        if lat is not None and lat != "" and lng is not None and lng != "":
            geo_scores = self.inferrer.geo_model.predict(lat, lng, iconic_taxon_id)
        else:
            geo_scores = {}
        geo_total_time = time.time() - geo_start_time
        print("GeoTime: %0.2fms" % (geo_total_time * 1000.))

        # Scoring
        scoring_start_time = time.time()
        combined_scores = ModelScoring.combine_vision_and_geo_scores(vision_scores, geo_scores)

        # results.aggregate_scores()
        scoring_total_time = time.time() - scoring_start_time
        print("Score Time: %0.2fms" % (scoring_total_time * 1000.))

        top_x = dict(sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True)[:100])
        to_return = []
        for index, arg in enumerate(top_x):
            geo_score = geo_scores[arg] if arg in geo_scores else 0.0000000001
            to_return.append({
                "combined_score": round(combined_scores[arg] * 100, 6),
                "vision_score": round(vision_scores[arg] * 100, 6),
                "geo_score": round(geo_score * 100, 6),
                "id": self.inferrer.taxonomy.taxa[arg].id,
                "name": self.inferrer.taxonomy.taxa[arg].name
            })

        total_time = time.time() - vision_start_time
        print("Total: %0.2fms" % (total_time * 1000.))
        return to_return

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

    def write_logstash(self, image_uuid, file_path, request_start_datetime, request_start_time):
        request_end_time = time.time()
        request_time = round((request_end_time - request_start_time) * 1000, 6)
        logstash_log = open('log/logstash.log', 'a')
        log_data = {'@timestamp': request_start_datetime.isoformat(),
                    'uuid': image_uuid,
                    'duration': request_time,
                    'client_ip': request.access_route[0],
                    'image_size': os.path.getsize(file_path)}
        json.dump(log_data, logstash_log)
        logstash_log.write("\n")
        logstash_log.close()
