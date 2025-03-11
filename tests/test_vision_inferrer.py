import tensorflow as tf
from unittest.mock import MagicMock
from lib.vision_inferrer_tf import VisionInferrerTF


class TestVisionInferrer:
    def test_initialization(self, mocker):
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        mocker.patch("tensorflow.keras.Model", return_value=MagicMock())
        model_path = "model_path"
        inferrer = VisionInferrerTF(model_path)
        assert inferrer.model_path == model_path
        tf.keras.models.load_model.assert_called_once_with(
            model_path,
            compile=False
        )

    def test_process_image(self, mocker):
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        mocker.patch("tensorflow.keras.Model", return_value=MagicMock())
        model_path = "model_path"
        inferrer = VisionInferrerTF(model_path)
        theimage = "theimage"
        inferrer.process_image(theimage)
        inferrer.layered_model.assert_called_once_with(
            tf.convert_to_tensor(theimage),
            training=False
        )
