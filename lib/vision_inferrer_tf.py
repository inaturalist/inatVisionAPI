import tensorflow as tf

from lib.vision_inferrer import VisionInferrer, VisionResults


class VisionInferrerTF(VisionInferrer):
    """Vision Inferrer for the TF variant of iNat vision models.
    Our implementation expects inputs in the range [0, 1).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.prepare_model()

    def prepare_model(self):
        """initialize the TF model given the configured path"""
        # disable GPU processing
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"

        full_model = tf.keras.models.load_model(self.model_path, compile=False)
        self.layered_model = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=[full_model.layers[-1].output, full_model.layers[2].output],
        )
        self.layered_model.compile()

    def process_image(self, image: tf.Tensor) -> VisionResults:
        """given an image object (coming from prepare_image_for_inference),
        calculate & return vision results for the image."""
        layer_results = self.layered_model(tf.convert_to_tensor(image), training=False)
        return {
            "predictions": layer_results[0][0],
            "features": layer_results[1][0],
        }
