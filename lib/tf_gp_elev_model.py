import tensorflow as tf
import numpy as np
import math
import os
from lib.res_layer import ResLayer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TFGeoPriorModelElev:

    def __init__(self, model_path):
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

    def predict(self, latitude, longitude, elevation):
        encoded_loc = TFGeoPriorModelElev.encode_loc([latitude], [longitude], [elevation])
        return self.gpmodel(tf.convert_to_tensor(
            tf.expand_dims(encoded_loc[0], axis=0)
        ), training=False)[0]

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
        return tf.keras.activations.sigmoid(
            tf.matmul(
                tf.expand_dims(self.gpmodel.layers[5].weights[0][:, class_of_interest], axis=0),
                features,
                transpose_b=True
            )
        ).numpy()

    @staticmethod
    def encode_loc(latitude, longitude, elevation):
        latitude = np.array(latitude)
        longitude = np.array(longitude)
        elevation = np.array(elevation)
        elevation = elevation.astype("float32")
        grid_lon = longitude.astype("float32") / 180.0
        grid_lat = latitude.astype("float32") / 90.0

        elevation[elevation > 0] = elevation[elevation > 0] / 6574.0
        elevation[elevation < 0] = elevation[elevation < 0] / 32768.0
        norm_elev = elevation

        # if np.isscalar(grid_lon):
        #     grid_lon = np.array([grid_lon])
        # if np.isscalar(grid_lat):
        #     grid_lat = np.array([grid_lat])
        # if np.isscalar(norm_elev):
        #     norm_elev = np.array([norm_elev])

        norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

        encoded_loc = tf.concat([
            tf.sin(norm_loc * math.pi),
            tf.cos(norm_loc * math.pi),
            tf.expand_dims(norm_elev, axis=1),

        ], axis=1)
        return encoded_loc
