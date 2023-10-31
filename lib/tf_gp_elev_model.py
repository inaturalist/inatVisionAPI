import tensorflow as tf
import numpy as np
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ResLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.w1 = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal"
        )
        self.w2 = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal"
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.w1(inputs)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.add([x, inputs])
        return x

    def get_config(self):
        return {}


class TFGeoPriorModelElev:

    def __init__(self, model_path):
        # initialize the geo model for inference
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
        self.gpmodel = tf.keras.models.load_model(
            model_path,
            custom_objects={'ResLayer': ResLayer},
            compile=False
        )

    def predict(self, latitude, longitude, elevation):
        norm_lat = latitude / 90.0
        norm_lng = longitude / 180.0
        norm_loc = tf.stack([norm_lng, norm_lat])

        if elevation > 0:
            norm_elev = elevation / 6574
        elif elevation == 0:
            norm_elev = 0.0
        else:
            norm_elev = elevation / 32768

        norm_elev = tf.expand_dims(norm_elev, axis=0)
        encoded_loc = tf.concat([
            tf.sin(norm_loc * math.pi),
            tf.cos(norm_loc * math.pi),
            norm_elev
        ], axis=0)

        return self.gpmodel(tf.convert_to_tensor(
            tf.expand_dims(encoded_loc, axis=0)
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
        def encode_loc(latitude, longitude, elevation):
            latitude = np.array(latitude)
            longitude = np.array(longitude)
            elevation = np.array(elevation)
            elevation = elevation.astype("float32")
            grid_lon = longitude.astype('float32') / 180.0
            grid_lat = latitude.astype('float32') / 90.0

            elevation[elevation > 0] = elevation[elevation > 0] / 6574.0
            elevation[elevation < 0] = elevation[elevation < 0] / 32768.0
            norm_elev = elevation

            if np.isscalar(grid_lon):
                grid_lon = np.array([grid_lon])
            if np.isscalar(grid_lat):
                grid_lat = np.array([grid_lat])
            if np.isscalar(norm_elev):
                norm_elev = np.array([norm_elev])

            norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

            encoded_loc = tf.concat([
                tf.sin(norm_loc * math.pi),
                tf.cos(norm_loc * math.pi),
                tf.expand_dims(norm_elev, axis=1),

            ], axis=1)
            return encoded_loc

        encoded_loc = encode_loc(latitude, longitude, elevation)
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
