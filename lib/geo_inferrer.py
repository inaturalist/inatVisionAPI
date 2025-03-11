from abc import ABC, abstractmethod
import math

import numpy as np
import tensorflow as tf


class GeoInferrer(ABC):
    @abstractmethod
    def __init__(self, model_path: str):
        """Subclasses must implement this constructor."""
        pass

    @abstractmethod
    def predict(
        self, latitude: float, longitude: float, elevation: float
    ) -> np.ndarray:
        """
        given a location, calculate geo results

        Subclasses must implement this method.
        """
        pass

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

        norm_loc = tf.stack([grid_lon, grid_lat], axis=1)

        encoded_loc = tf.concat(
            [
                tf.sin(norm_loc * math.pi),
                tf.cos(norm_loc * math.pi),
                tf.expand_dims(norm_elev, axis=1),
            ],
            axis=1,
        )
        return encoded_loc
