import os
import magic
import tensorflow as tf
import pandas as pd
import h3
from PIL import Image
from lib.pt_geo_prior_model import PTGeoPriorModel
from lib.tf_gp_model import TFGeoPriorModel
from lib.tf_gp_elev_model import TFGeoPriorModelElev
from lib.vision_inferrer import VisionInferrer
from lib.model_taxonomy import ModelTaxonomy


class InatInferrer:

    def __init__(self, config):
        self.config = config
        self.setup_taxonomy(config)
        self.setup_vision_model(config)
        self.setup_elevation_dataframe(config)
        self.setup_geo_thresholds(config)
        self.setup_geo_model(config)
        self.upload_folder = "static/"

    def setup_taxonomy(self, config):
        self.taxonomy = ModelTaxonomy(config["taxonomy_path"])

    def setup_vision_model(self, config):
        self.vision_inferrer = VisionInferrer(config["vision_model_path"], self.taxonomy)

    def setup_elevation_dataframe(self, config):
        # load elevation data stored at H3 resolution 4
        if "elevation_h3_r4" in config:
            self.geo_elevation_cells = pd.read_csv(config["elevation_h3_r4"]). \
                sort_values("h3_04").set_index("h3_04").sort_index()

    def setup_geo_thresholds(self, config):
        if "tf_elev_thresholds" in config:
            self.geo_thresholds = pd.read_csv(config["tf_elev_thresholds"]). \
                iloc[:, 1:].set_index("taxon_id").sort_index()
            self.geo_thresholds["thres"] = self.geo_thresholds["thres"].multiply(100)

    def setup_geo_model(self, config):
        if "use_pt_gp_model" in config and config["use_pt_gp_model"] and "pt_geo_model_path" in config:
            self.geo_model = PTGeoPriorModel(config["pt_geo_model_path"], self.taxonomy)
        elif "tf_geo_model_path" in config:
            self.geo_model = TFGeoPriorModel(config["tf_geo_model_path"], self.taxonomy)
        else:
            self.geo_model = None
        if "tf_geo_elevation_model_path" in config and self.geo_elevation_cells is not None:
            self.geo_elevation_model = TFGeoPriorModelElev(config["tf_geo_elevation_model_path"], self.taxonomy)

    def prepare_image_for_inference(self, file_path, image_uuid):
        mime_type = magic.from_file(file_path, mime=True)
        # attempt to convert non jpegs
        if mime_type != "image/jpeg":
            im = Image.open(file_path)
            rgb_im = im.convert("RGB")
            file_path = os.path.join(self.upload_folder, image_uuid) + ".jpg"
            rgb_im.save(file_path)

        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.central_crop(image, 0.875)
        image = tf.image.resize(image, [299, 299], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.expand_dims(image, 0)

    def vision_predict(self, image, iconic_taxon_id):
        return self.vision_inferrer.process_image(image, iconic_taxon_id)

    def geo_model_predict(self, lat, lng, iconic_taxon_id, geo_model_type="original"):
        if lat is None or lat == "" or lng is None or lng == "":
            return {}
        if geo_model_type == "elevation" and self.geo_elevation_model is None:
            return {}
        if geo_model_type == "elevation":
            # lookup the H3 cell this lat lng occurs in
            h3_cell = h3.latlng_to_cell(float(lat), float(lng), 4)
            # get the average elevation of the above H3 cell
            elevation = self.geo_elevation_cells.loc[h3_cell].elevation
            geo_scores = self.geo_elevation_model.predict(float(lat), float(lng), float(elevation), iconic_taxon_id)
        else:
            geo_scores = self.geo_model.predict(lat, lng, iconic_taxon_id)
        return geo_scores

    def is_seen_nearby(self, taxon_id, geo_score):
        return True if geo_score >= self.geo_thresholds.loc[taxon_id].thres else False
