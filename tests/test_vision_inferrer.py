import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock
from lib.vision_inferrer import VisionInferrer


class TestVisionInferrer:
    def test_initialization(self, mocker, mock_model):
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        mocker.patch("tensorflow.keras.Model", return_value=mock_model)
        model_path = "model_path"
        inferrer = VisionInferrer(model_path)
        assert inferrer.model_path == model_path
        tf.keras.models.load_model.assert_called_once_with(
            model_path,
            compile=False
        )

    def test_process_image(self, mocker, mock_model):
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        mocker.patch("tensorflow.keras.Model", return_value=mock_model)
        model_path = "model_path"
        inferrer = VisionInferrer(model_path)
        theimage = tf.zeros(
            shape=(1, 299, 299, 3),
            dtype=tf.float32,
        )
        results = inferrer.process_image(theimage)

        np.testing.assert_allclose(
            results["predictions"].numpy(),
            [0.2, 0.3, 0.5],
        )
        np.testing.assert_allclose(
            results["features"].numpy(),
            [0.0],
        )

        inferrer.layered_model.assert_called_once()
        args, kwargs = inferrer.layered_model.call_args
        traced_image = args[0]
        assert tf.is_tensor(traced_image)
        assert traced_image.shape == tf.TensorShape([1, 299, 299, 3])
        assert traced_image.dtype == tf.float32
        assert kwargs == {"training": False}
