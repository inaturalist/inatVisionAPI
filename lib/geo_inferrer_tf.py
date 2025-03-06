import os

import tensorflow as tf
import numpy as np

from lib.res_layer import ResLayer
from lib.geo_inferrer import GeoInferrer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TFGeoPriorModelElev(GeoInferrer):

    def __init__(self, model_path: str):
        # initialize the geo model for inference
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"
        self.gpmodel = tf.keras.models.load_model(
            model_path,
            custom_objects={"ResLayer": ResLayer}, 
            compile=False
        )

    def predict(self, latitude: float, longitude: float, elevation: float) -> np.ndarray:
        encoded_loc = GeoInferrer.encode_loc([latitude], [longitude], [elevation])
        output = self.gpmodel(tf.convert_to_tensor(
            tf.expand_dims(encoded_loc[0], axis=0)
        ), training=False)[0]
        return output

    def features_for_one_class_elevation(self, latitude, longitude, elevation):
        """Evalutes the model for a single class and multiple locations

        Args:
            latitude (list): A list of latitudes
            longitude (list): A list of longitudes (same length as latitude)
            elevation (list): A list of elevations (same length as latitude)
            class_of_interest (int): The single class to eval

        Returns:
            numpy array: scores for class of interest at each location
        """

        encoded_loc = TFGeoPriorModelElev.encode_loc(latitude, longitude, elevation)
        loc_emb = self.gpmodel.layers[0](encoded_loc)

        # res layers - feature extraction
        x = self.gpmodel.layers[1](loc_emb)
        x = self.gpmodel.layers[2](x)
        x = self.gpmodel.layers[3](x)
        x = self.gpmodel.layers[4](x)

        return x

    def eval_one_class_elevation_from_features(self, features, class_of_interest):
        # process just the one class
        return tf.math.sigmoid(
            tf.matmul(
                tf.expand_dims(self.gpmodel.layers[5].weights[0][:, class_of_interest], axis=0),
                features,
                transpose_b=True
            )
        ).numpy()
