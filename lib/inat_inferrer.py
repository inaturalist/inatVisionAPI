import os
import magic
import tensorflow as tf
from PIL import Image
from lib.geo_prior_model import GeoPriorModel
from lib.tf_gp_model import TFGeoPriorModel
from lib.vision_inferrer import VisionInferrer
from lib.model_taxonomy import ModelTaxonomy


class InatInferrer:

    def __init__(self, config):
        self.config = config
        self.setup_taxonomy(config)
        self.setup_vision_model(config)
        self.setup_geo_model(config)
        self.upload_folder = "static/"

    def setup_taxonomy(self, config):
        self.taxonomy = ModelTaxonomy(config["taxonomy_path"])

    def setup_vision_model(self, config):
        self.vision_inferrer = VisionInferrer(config["vision_model_path"], self.taxonomy)

    def setup_geo_model(self, config):
        if "use_tf_gp_model" in config and config["use_tf_gp_model"] and "geo_model_path" in config:
            self.geo_model = TFGeoPriorModel(config["tf_geo_model_path"], self.taxonomy)
        elif "geo_model_path" in config:
            self.geo_model = GeoPriorModel(config["geo_model_path"], self.taxonomy)
        else:
            self.geo_model = None

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
