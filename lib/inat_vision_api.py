import datetime
import time
import os
import urllib
import uuid
import json

from flask import Flask, request, render_template
from web_forms import ImageForm
from inat_inferrer import InatInferrer
from inat_vision_api_responses import InatVisionAPIResponses


class InatVisionAPI:

    def __init__(self, config):
        self.setup_inferrer(config["models"][0])
        self.app = Flask(__name__)
        self.app.secret_key = config["app_secret"]
        self.upload_folder = "static/"
        self.app.add_url_rule("/", "index", self.index_route, methods=["GET", "POST"])
        self.app.add_url_rule("/refresh_synonyms", "refresh_synonyms",
                              self.refresh_synonyms, methods=["POST"])
        self.app.add_url_rule("/h3_04", "h3_04", self.h3_04_route, methods=["GET"])
        self.app.add_url_rule("/h3_04_taxon_range", "h3_04_taxon_range",
                              self.h3_04_taxon_range_route, methods=["GET"])
        self.app.add_url_rule("/h3_04_taxon_range_comparison", "h3_04_taxon_range_comparison",
                              self.h3_04_taxon_range_comparison_route, methods=["GET"])
        self.app.add_url_rule("/h3_04_bounds", "h3_04_bounds",
                              self.h3_04_bounds_route, methods=["GET"])
        self.app.add_url_rule("/geo_scores_for_taxa", "geo_scores_for_taxa",
                              self.geo_scores_for_taxa_route, methods=["POST"])
        self.app.add_url_rule("/build_info", "build_info", self.build_info_route, methods=["GET"])

    def setup_inferrer(self, config):
        self.inferrer = InatInferrer(config)

    def refresh_synonyms(self):
        self.inferrer.refresh_synonyms_if_modified()
        return ("", 204)

    def h3_04_route(self):
        return self.h3_04_default_route(self.inferrer.h3_04_geo_results_for_taxon)

    def h3_04_taxon_range_route(self):
        if "taxon_ranges_path" not in self.inferrer.config:
            return "taxon range data unavilable", 422
        return self.h3_04_default_route(self.inferrer.h3_04_taxon_range)

    def h3_04_taxon_range_comparison_route(self):
        if "taxon_ranges_path" not in self.inferrer.config:
            return "taxon range data unavilable", 422
        return self.h3_04_default_route(self.inferrer.h3_04_taxon_range_comparison)

    def h3_04_default_route(self, h3_04_method):
        taxon_id, error_message, error_code = self.valid_leaf_taxon_id_for_request(request)
        if error_message:
            return error_message, error_code

        bounds, error_message, error_code = self.valid_bounds_for_request(request)
        if error_message:
            return error_message, error_code

        if h3_04_method == self.inferrer.h3_04_geo_results_for_taxon \
           and "thresholded" in request.args and request.args["thresholded"] == "true":
            results_dict = h3_04_method(taxon_id, bounds, thresholded=True)
        else:
            results_dict = h3_04_method(taxon_id, bounds)
        if results_dict is None:
            return f"Unknown taxon_id {taxon_id}", 422
        return InatVisionAPI.round_floats(results_dict, 8)

    def h3_04_bounds_route(self):
        taxon_id, error_message, error_code = self.valid_leaf_taxon_id_for_request(request)
        if error_message:
            return error_message, error_code

        results_dict = self.inferrer.h3_04_bounds(taxon_id)
        if results_dict is None:
            return f"Unknown taxon_id {taxon_id}", 422
        return results_dict

    def build_info_route(self):
        return {
            "git_branch": os.getenv("GIT_BRANCH", ""),
            "git_commit": os.getenv("GIT_COMMIT", ""),
            "image_tag": os.getenv("IMAGE_TAG", ""),
            "build_date": os.getenv("BUILD_DATE", "")
        }

    def geo_scores_for_taxa_route(self):
        return {
            obs["id"]: self.inferrer.h3_04_geo_results_for_taxon_and_cell(
                obs["taxon_id"], obs["lat"], obs["lng"]
            )
            for obs in request.json["observations"]
        }

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
            iconic_taxon_id = None
            if observation_id:
                image_uuid = "downloaded-obs-" + observation_id
                file_path, lat, lng, iconic_taxon_id = self.download_observation(
                    observation_id, image_uuid)
            else:
                image_uuid = str(uuid.uuid4())
                file_path = self.process_upload(form.image.data, image_uuid)
                if form.taxon_id.data and form.taxon_id.data.isdigit():
                    iconic_taxon_id = int(form.taxon_id.data)
            if file_path is None:
                return render_template("home.html")

            scores = self.score_image(form, file_path, lat, lng, iconic_taxon_id, geomodel)
            InatVisionAPI.write_logstash(
                image_uuid, file_path, request_start_datetime, request_start_time)
            return scores
        else:
            return render_template("home.html")

    def score_image(self, form, file_path, lat, lng, iconic_taxon_id, geomodel):
        filter_taxon = self.inferrer.lookup_taxon(iconic_taxon_id)
        leaf_scores = self.inferrer.predictions_for_image(
            file_path, lat, lng, filter_taxon, debug=True
        )

        if form.aggregated.data == "true":
            aggregated_scores = self.inferrer.aggregate_results(leaf_scores, debug=True)
            if form.format.data == "tree":
                return InatVisionAPIResponses.aggregated_tree_response(
                    aggregated_scores, self.inferrer
                )
            return InatVisionAPIResponses.aggregated_object_response(
                leaf_scores, aggregated_scores, self.inferrer
            )

        # legacy dict response
        if geomodel != "true":
            return InatVisionAPIResponses.legacy_dictionary_response(leaf_scores, self.inferrer)

        if form.format.data == "object":
            return InatVisionAPIResponses.object_response(leaf_scores, self.inferrer)

        return InatVisionAPIResponses.array_response(leaf_scores, self.inferrer)

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
        if data["results"][0]["location"] is None:
            latlng = [None, None]
        else:
            latlng = data["results"][0]["location"].split(",")
        # return the path to the cached image, coordinates, and iconic taxon
        return cache_path, latlng[0], latlng[1], data["results"][0]["taxon"]["iconic_taxon_id"]

    def valid_leaf_taxon_id_for_request(self, request):
        if "taxon_id" in request.args:
            taxon_id = request.args["taxon_id"]
        else:
            return None, "taxon_id required", 422
        if not taxon_id.isdigit():
            return None, "taxon_id must be an integer", 422

        taxon_id = int(taxon_id)
        if float(taxon_id) not in self.inferrer.taxonomy.leaf_df["taxon_id"].values:
            return None, f"Unknown taxon_id {taxon_id}", 422
        return taxon_id, None, None

    def valid_bounds_for_request(self, request):
        bounds = []
        if "swlat" in request.args:
            try:
                swlat = float(request.args["swlat"])
                swlng = float(request.args["swlng"])
                nelat = float(request.args["nelat"])
                nelng = float(request.args["nelng"])
            except ValueError:
                return None, "bounds must be floats", 422
            bounds = [swlat, swlng, nelat, nelng]
        return bounds, None, None

    @staticmethod
    def write_logstash(image_uuid, file_path, request_start_datetime, request_start_time):
        request_end_time = time.time()
        request_time = round((request_end_time - request_start_time) * 1000, 6)
        logstash_log = open("log/logstash.log", "a")
        log_data = {"@timestamp": request_start_datetime.isoformat(),
                    "uuid": image_uuid,
                    "duration": request_time,
                    "client_ip": request.access_route[0],
                    "image_size": os.path.getsize(file_path)}
        json.dump(log_data, logstash_log)
        logstash_log.write("\n")
        logstash_log.close()

    @staticmethod
    def round_floats(o, sig):
        if isinstance(o, float):
            return round(o, sig)
        if isinstance(o, dict):
            return {k: InatVisionAPI.round_floats(v, sig) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [InatVisionAPI.round_floats(x, sig) for x in o]
        return o
