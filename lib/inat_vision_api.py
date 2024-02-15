import datetime
import time
import os
import urllib
import uuid
import json

from flask import Flask, request, render_template
from web_forms import ImageForm
from inat_inferrer import InatInferrer


class InatVisionAPI:

    def __init__(self, config):
        self.setup_inferrer(config["models"][0])
        self.app = Flask(__name__)
        self.app.secret_key = config["app_secret"]
        self.upload_folder = "static/"
        self.app.add_url_rule("/", "index", self.index_route, methods=["GET", "POST"])
        self.app.add_url_rule("/h3_04", "h3_04", self.h3_04_route, methods=["GET"])
        self.app.add_url_rule("/h3_04_taxon_range", "h3_04_taxon_range",
                              self.h3_04_taxon_range_route, methods=["GET"])
        self.app.add_url_rule("/h3_04_taxon_range_comparison", "h3_04_taxon_range_comparison",
                              self.h3_04_taxon_range_comparison_route, methods=["GET"])
        self.app.add_url_rule("/h3_04_bounds", "h3_04_bounds",
                              self.h3_04_bounds_route, methods=["GET"])

    def setup_inferrer(self, config):
        self.inferrer = InatInferrer(config)

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
            return f'Unknown taxon_id {taxon_id}', 422
        return InatVisionAPI.round_floats(results_dict, 8)

    def h3_04_bounds_route(self):
        taxon_id, error_message, error_code = self.valid_leaf_taxon_id_for_request(request)
        if error_message:
            return error_message, error_code

        results_dict = self.inferrer.h3_04_bounds(taxon_id)
        if results_dict is None:
            return f'Unknown taxon_id {taxon_id}', 422
        return results_dict

    def bench_route(self):
        start = request.args["start"]
        count = request.args["count"]
        self.inferrer.TIME_DOWNLOAD = 0
        self.inferrer.TIME_RESIZE = 0
        self.inferrer.TIME_TOTAL = 0
        
        ALL_SCORES = ""

        for observation_id in range(start, start + count): 
            print("TIME-EXP: score "+observation_id)
            START_TIME_TOTAL = time.time()
            image_uuid = "downloaded-obs-" + observation_id
            START_TIME_DOWNLOAD = time.time()
            file_path, lat, lng, iconic_taxon_id = self.download_observation(
                observation_id, image_uuid)
            END_TIME_DOWNLOAD = time.time()
            self.inferrer.TIME_DOWNLOAD = self.inferrer.TIME_DOWNLOAD + (END_TIME_DOWNLOAD - START_TIME_DOWNLOAD)
            print("TIME-EXP: TIME_DOWNLOAD "+str(self.inferrer.TIME_DOWNLOAD))
            scores = self.score_image(form, file_path, lat, lng, iconic_taxon_id, geomodel)
            END_TIME_TOTAL = time.time()
            self.inferrer.TIME_TOTAL = self.inferrer.TIME_TOTAL + (END_TIME_TOTAL - START_TIME_TOTAL)
            print("TIME-EXP: TIME_TOTAL "+str(self.inferrer.TIME_TOTAL))
            ALL_SCORES = ALL_SCORES + "\n" + scores

        result = "TOTAL = " + str(self.inferrer.TIME_TOTAL) + "\n" + \
                 "DOWNLOAD = " + str(self.inferrer.TIME_DOWNLOAD) + "\n" + \
                 "RESIZE = " + str(self.inferrer.TIME_RESIZE) + "\n" + \
                 ALL_SCORES

        return result

    def index_route(self):
        START_TIME_TOTAL = time.time()
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
                START_TIME_DOWNLOAD = time.time()
                file_path, lat, lng, iconic_taxon_id = self.download_observation(
                    observation_id, image_uuid)
                END_TIME_DOWNLOAD = time.time()
                self.inferrer.TIME_DOWNLOAD = self.inferrer.TIME_DOWNLOAD + (END_TIME_DOWNLOAD - START_TIME_DOWNLOAD)
            else:
                image_uuid = str(uuid.uuid4())
                file_path = self.process_upload(form.image.data, image_uuid)
                if form.taxon_id.data and form.taxon_id.data.isdigit():
                    iconic_taxon_id = int(form.taxon_id.data)
            if file_path is None:
                return render_template("home.html")

            scores = self.score_image(form, file_path, lat, lng, iconic_taxon_id, geomodel)
            END_TIME_TOTAL = time.time()
            self.inferrer.TIME_TOTAL = self.inferrer.TIME_TOTAL + (END_TIME_TOTAL - START_TIME_TOTAL)
            InatVisionAPI.write_logstash(
                image_uuid, file_path, request_start_datetime, request_start_time)
            return scores
        else:
            return render_template("home.html")

    def score_image(self, form, file_path, lat, lng, iconic_taxon_id, geomodel):
        score_without_geo = (form.score_without_geo.data == "true")
        filter_taxon = self.inferrer.lookup_taxon(iconic_taxon_id)
        leaf_scores = self.inferrer.predictions_for_image(
            file_path, lat, lng, filter_taxon, score_without_geo
        )

        if form.aggregated.data == "true":
            aggregated_results = self.inferrer.aggregate_results(leaf_scores, filter_taxon,
                                                                 score_without_geo)
            columns_to_return = [
                "aggregated_combined_score",
                "aggregated_geo_score",
                "taxon_id",
                "name",
                "aggregated_vision_score",
                "aggregated_geo_threshold"
            ]
            column_mapping = {
                "taxon_id": "id",
                "aggregated_combined_score": "combined_score",
                "aggregated_geo_score": "geo_score",
                "aggregated_vision_score": "vision_score",
                "aggregated_geo_threshold": "geo_threshold"
            }

            no_geo_scores = (leaf_scores["geo_score"].max() == 0)

            # set a cutoff where branches whose combined scores are below the threshold are ignored
            # TODO: this threshold is completely arbitrary and needs testing
            aggregated_results = aggregated_results.query("normalized_aggregated_combined_score > 0.05")

            # after setting a cutoff, get the parent IDs of the remaining taxa
            parent_taxon_ids = aggregated_results["parent_taxon_id"].values  # noqa: F841
            # the leaves of the pruned taxonomy (not leaves of the original taxonomy), are the
            # taxa who are not parents of any remaining taxa
            leaf_results = aggregated_results.query("taxon_id not in @parent_taxon_ids")

            leaf_results = leaf_results.sort_values("aggregated_combined_score", ascending=False).head(100)
            score_columns = ["aggregated_combined_score", "aggregated_geo_score",
                             "aggregated_vision_score", "aggregated_geo_threshold"]
            leaf_results[score_columns] = leaf_results[score_columns].multiply(100, axis="index")
            final_results = leaf_results[columns_to_return].rename(columns=column_mapping)
        else:
            no_geo_scores = (leaf_scores["geo_score"].max() == 0)
            top_combined_score = leaf_scores.sort_values("combined_score", ascending=False).head(1)["combined_score"].values[0]
            # set a cutoff so results whose combined scores are
            # much lower than the best score are not returned
            leaf_scores = leaf_scores.query(f'combined_score > {top_combined_score * 0.001}')

            top100 = leaf_scores.sort_values("combined_score", ascending=False).head(100)
            score_columns = ["combined_score", "geo_score", "normalized_vision_score", "geo_threshold"]
            top100[score_columns] = top100[score_columns].multiply(100, axis="index")

            # legacy dict response
            if geomodel != "true":
                top_taxon_combined_scores = top100[
                    ["taxon_id", "combined_score"]
                ].to_dict(orient="records")
                return {x["taxon_id"]: x["combined_score"] for x in top_taxon_combined_scores}

            # new array response
            columns_to_return = [
                "combined_score",
                "geo_score",
                "taxon_id",
                "name",
                "normalized_vision_score",
                "geo_threshold"
            ]
            column_mapping = {
                "taxon_id": "id",
                "normalized_vision_score": "vision_score"
            }
            final_results = top100[columns_to_return].rename(columns=column_mapping)

        return final_results.to_dict(orient="records")

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

    def valid_leaf_taxon_id_for_request(self, request):
        if "taxon_id" in request.args:
            taxon_id = request.args["taxon_id"]
        else:
            return None, "taxon_id required", 422
        if not taxon_id.isdigit():
            return None, "taxon_id must be an integer", 422

        taxon_id = int(taxon_id)
        if float(taxon_id) not in self.inferrer.taxonomy.leaf_df["taxon_id"].values:
            return None, f'Unknown taxon_id {taxon_id}', 422
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
        logstash_log = open('log/logstash.log', 'a')
        log_data = {'@timestamp': request_start_datetime.isoformat(),
                    'uuid': image_uuid,
                    'duration': request_time,
                    'client_ip': request.access_route[0],
                    'image_size': os.path.getsize(file_path)}
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
