import pytest
import tensorflow as tf
from unittest.mock import MagicMock
import unittest.mock as mock

from lib.res_layer import ResLayer
from lib.geo_inferrer_tf import TFGeoPriorModelElev


class TestTfGpModel:
    def test_initialization_with_unknown_model_path(self):
        with pytest.raises(OSError):
            TFGeoPriorModelElev("model_path")

    def test_initialization(self, mocker):
        model_path = "model_path"
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        TFGeoPriorModelElev(model_path)
        tf.keras.models.load_model.assert_called_once_with(
            model_path, custom_objects={"ResLayer": ResLayer}, compile=False
        )

    def test_predict(self, mocker):
        model_path = "model_path"
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        tf_gp_model = TFGeoPriorModelElev(model_path)
        tf_gp_model.predict(0, 0, 0)

    def test_features_for_one_class_elevation(self, mocker):
        model_path = "model_path"
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        tf_gp_model = TFGeoPriorModelElev(model_path)
        tf_gp_model.features_for_one_class_elevation([0], [0], [0])

    def test_eval_one_class_elevation_from_features(self, mocker):
        tf.math.sigmoid = mock.create_autospec(tf.math.sigmoid)
        model_path = "model_path"
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        mocker.patch("tensorflow.keras.activations.sigmoid", return_value=MagicMock())
        mocker.patch("tensorflow.matmul", return_value=MagicMock())
        mocker.patch("tensorflow.expand_dims", return_value=MagicMock())
        tf_gp_model = TFGeoPriorModelElev(model_path)
        tf_gp_model.eval_one_class_elevation_from_features(
            "features", "class_of_interest"
        )
        tf.math.sigmoid.assert_called_once
