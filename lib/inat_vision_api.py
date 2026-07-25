import time
import os
import urllib
import uuid
import json

from flask import Flask, request, render_template, g
from web_forms import ImageForm
from annotation_inferrer import AnnotationInferrer
from inat_inferrer import InatInferrer
from inat_vision_api_responses import InatVisionAPIResponses
from logstasher import Logstasher
from model_taxonomy_dataframe import ModelTaxonomyDataframe


class InatVisionAPI:

    def __init__(self, config):
        self.debug = config["debug"] if "debug" in config else False
        self.setup_inferrer(config["models"][0])
        self.app = Flask(__name__)
        self.app.secret_key = config["app_secret"]
        self.upload_folder = "static/"
        self.logstasher = Logstasher()
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
        self.app.add_url_rule("/embeddings_for_photos", "embeddings_for_photos",
                              self.embeddings_for_photos_route, methods=["POST"])
        self.app.add_url_rule("/build_info", "build_info", self.build_info_route, methods=["GET"])
        self.app.before_request(self.before_request)
        self.app.after_request(self.after_request)

    def before_request(self):
        g.request_start_time = time.time()

    def after_request(self, response):
        self.logstasher.log_request(request, response, g)
        return response

    def setup_inferrer(self, config):
        self.inferrer = InatInferrer(config)
        self.setup_annotation_inferrer(config)

    def setup_annotation_inferrer(self, config):
        self.annotation_inferrer = None
        model_dir = config.get("annotation_model_path")
        if AnnotationInferrer.bundle_is_present(model_dir):
            self.annotation_inferrer = AnnotationInferrer(model_dir, self.inferrer.taxonomy)
            print(
                f"Annotation heads loaded from {model_dir} "
                f"(run {self.annotation_inferrer.run})"
            )
        elif model_dir:
            print(f"Annotation heads NOT loaded: {model_dir} is missing bundle files")
        else:
            print("Annotation heads NOT loaded: no annotation_model_path configured")

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

    async def embeddings_for_photos_route(self):
        start_time = time.time()
        response = await self.inferrer.embeddings_for_photos(request.json["photos"])
        print("embeddings_for_photos_route Time: %0.2fms" % ((time.time() - start_time) * 1000.))
        return response

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
            lat = form.lat.data
            lng = form.lng.data
            common_ancestor_rank_type = form.common_ancestor_rank_type.data
            file_path = None
            image_uuid = None
            filter_taxon_id = None
            observation_taxon_id = None
            observation_ancestor_ids = None

            annotations_requested = InatVisionAPI.annotations_requested()
            if annotations_requested and self.annotation_inferrer is None:
                return "annotation model bundle not configured", 422
            override_taxon_id, error_message, error_code = \
                InatVisionAPI.annotation_taxon_id_for_request(form)
            if error_message:
                return error_message, error_code
            override_ancestor_ids, error_message, error_code = \
                InatVisionAPI.annotation_ancestry_for_request(form)
            if error_message:
                return error_message, error_code

            if observation_id:
                image_uuid = "downloaded-obs-" + observation_id
                file_path, lat, lng, filter_taxon_id, observation_taxon_id, \
                    observation_ancestor_ids = self.download_observation(
                        observation_id, image_uuid)
            else:
                image_uuid = str(uuid.uuid4())
                file_path = self.process_upload(form.image.data, image_uuid)
                if form.taxon_id.data and form.taxon_id.data.isdigit():
                    filter_taxon_id = int(form.taxon_id.data)
            if file_path is None:
                return render_template("home.html")

            annotation_taxon = InatVisionAPI.annotation_taxon_for_request(
                override_taxon_id, override_ancestor_ids,
                observation_taxon_id, observation_ancestor_ids
            )
            scores = self.score_image(form, file_path, lat, lng, filter_taxon_id, geomodel,
                                      common_ancestor_rank_type,
                                      annotations_requested=annotations_requested,
                                      annotation_taxon=annotation_taxon)
            g.image_uuid = image_uuid
            g.image_size = os.path.getsize(file_path)
            return scores
        else:
            return render_template("home.html")

    def score_image(self, form, file_path, lat, lng, filter_taxon_id, geomodel,
                    common_ancestor_rank_type=None, annotations_requested=False,
                    annotation_taxon=None):
        if annotations_requested and not InatVisionAPI.format_supports_annotations(form, geomodel):
            return "annotations require format=object, and geomodel=true unless aggregated", 422

        filter_taxon = None
        if filter_taxon_id is not None:
            if self.inferrer.taxon_exists(filter_taxon_id):
                filter_taxon = self.inferrer.lookup_taxon(filter_taxon_id)
            else:
                filter_taxon = ModelTaxonomyDataframe.undefined_filter_taxon()
        predictions_for_image = self.inferrer.predictions_for_image(
            file_path, lat, lng, filter_taxon, debug=self.debug
        )
        leaf_scores = predictions_for_image["combined_scores"]

        annotations = None
        if annotations_requested:
            annotations = self.annotations_for_predictions(
                predictions_for_image, annotation_taxon,
                masking=(form.annotation_masking.data != "false")
            )

        if form.aggregated.data == "true":
            aggregated_scores = self.inferrer.aggregate_results(leaf_scores, debug=self.debug)
            if form.format.data == "tree":
                return InatVisionAPIResponses.aggregated_tree_response(
                    aggregated_scores, self.inferrer
                )
            embedding = predictions_for_image["features"]
            return InatVisionAPIResponses.aggregated_object_response(
                leaf_scores, aggregated_scores, self.inferrer,
                embedding=embedding,
                human_exclusion_strategy=form.human_exclusion.data,
                annotations=annotations
            )

        # legacy dict response
        if geomodel != "true":
            return InatVisionAPIResponses.legacy_dictionary_response(leaf_scores, self.inferrer)

        if form.format.data == "object":
            embedding = predictions_for_image["features"]
            return InatVisionAPIResponses.object_response(
                leaf_scores,
                self.inferrer,
                common_ancestor_rank_type=common_ancestor_rank_type,
                embedding=embedding,
                human_exclusion_strategy=form.human_exclusion.data,
                debug=self.debug,
                annotations=annotations
            )

        return InatVisionAPIResponses.array_response(leaf_scores, self.inferrer)

    def annotations_for_predictions(self, predictions_for_image, annotation_taxon, masking=True):
        taxon_id, ancestor_ids, taxon_source = annotation_taxon
        if taxon_id is None:
            taxon_id = InatVisionAPI.top_taxon_id(predictions_for_image["combined_scores"])
            taxon_source = "vision_top_result" if taxon_id is not None else "none"
        annotations = self.annotation_inferrer.predict(
            predictions_for_image["features"], taxon_id,
            ancestor_ids=ancestor_ids, mask_by_applicability=masking
        )
        annotations["taxon_source"] = taxon_source
        return annotations

    @staticmethod
    def top_taxon_id(leaf_scores):
        if leaf_scores.empty:
            return None
        return int(leaf_scores.loc[leaf_scores["combined_score"].idxmax()]["taxon_id"])

    @staticmethod
    def format_supports_annotations(form, geomodel):
        if form.format.data != "object":
            return False
        return form.aggregated.data == "true" or geomodel == "true"

    @staticmethod
    def annotations_requested():
        requested = request.args.get("annotations") or request.form.get("annotations")
        return requested == "true"

    @staticmethod
    def annotation_taxon_for_request(override_taxon_id, override_ancestor_ids,
                                     observation_taxon_id, observation_ancestor_ids):
        if override_taxon_id is not None:
            return override_taxon_id, override_ancestor_ids, "provided"
        if observation_taxon_id is not None:
            return observation_taxon_id, observation_ancestor_ids, "observation"
        return None, None, "vision_top_result"

    @staticmethod
    def annotation_taxon_id_for_request(form):
        raw_taxon_id = request.args.get("annotation_taxon_id") or form.annotation_taxon_id.data
        if not raw_taxon_id or not raw_taxon_id.strip():
            return None, None, None
        raw_taxon_id = raw_taxon_id.strip()
        if not raw_taxon_id.isdigit():
            return None, "annotation_taxon_id must be an integer", 422
        return int(raw_taxon_id), None, None

    @staticmethod
    def annotation_ancestry_for_request(form):
        raw_ancestry = request.args.get("annotation_taxon_ancestry") \
            or form.annotation_taxon_ancestry.data
        if not raw_ancestry or not raw_ancestry.strip():
            return None, None, None
        ancestor_ids = []
        for part in raw_ancestry.replace("/", ",").replace(" ", "").strip("[]").split(","):
            if not part:
                continue
            if not part.isdigit():
                return None, f"annotation_taxon_ancestry contains a non-integer: `{part}`", 422
            ancestor_ids.append(int(part))
        return ancestor_ids or None, None, None

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
            return None, None, None, None, None, None
        # download the first image if it isn't already cached
        if not os.path.exists(cache_path):
            urllib.request.urlretrieve(
                data["results"][0]["photos"][0]["url"].replace("square", "medium"), cache_path)
        if data["results"][0]["location"] is None:
            latlng = [None, None]
        else:
            latlng = data["results"][0]["location"].split(",")
        taxon = data["results"][0].get("taxon") or {}
        # return the path to the cached image, coordinates, and iconic taxon
        return cache_path, latlng[0], latlng[1], taxon.get("iconic_taxon_id"), \
            taxon.get("id"), taxon.get("ancestor_ids")

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
    def round_floats(o, sig):
        if isinstance(o, float):
            return round(o, sig)
        if isinstance(o, dict):
            return {k: InatVisionAPI.round_floats(v, sig) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [InatVisionAPI.round_floats(x, sig) for x in o]
        return o
