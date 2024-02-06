import tensorflow as tf
import unittest.mock as mock
from lib.res_layer import ResLayer
from unittest.mock import MagicMock


class TestResLayer:
    def test_initialization(self):
        res_layer = ResLayer()
        assert isinstance(res_layer.w1, tf.keras.layers.Dense)
        assert isinstance(res_layer.w2, tf.keras.layers.Dense)
        assert isinstance(res_layer.dropout, tf.keras.layers.Dropout)
        assert isinstance(res_layer.add, tf.keras.layers.Add)

    def test_call(self, mocker):
        mocker.patch("tensorflow.keras.models.load_model", return_value=MagicMock())
        res_layer = ResLayer()
        inputs = tf.keras.Input((256,))
        res_layer.call(inputs)
        call_w1 = mock.create_autospec(res_layer.w1.call)
        call_dropout = mock.create_autospec(res_layer.dropout.call)
        call_w2 = mock.create_autospec(res_layer.w1.call)
        call_add = mock.create_autospec(res_layer.add.call)
        call_w1.assert_called_once
        call_dropout.assert_called_once
        call_w2.assert_called_once
        call_add.assert_called_once

    def test_get_config(self):
        res_layer = ResLayer()
        assert res_layer.get_config() == {}
