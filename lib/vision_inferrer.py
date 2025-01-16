import tensorflow as tf


class VisionInferrer:

    def __init__(self, model_path):
        self.model_path = model_path
        self.prepare_tf_model()

    # initialize the TF model given the configured path
    def prepare_tf_model(self):
        # disable GPU processing
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"

        full_model = tf.keras.models.load_model(self.model_path, compile=False)
        self.layered_model = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=[
                full_model.layers[4].output,
                full_model.layers[2].output
            ]
        )
        self.layered_model.compile()

    # given an image object (usually coming from prepare_image_for_inference),
    # calculate vision results for the image
    def process_image(self, image):
        layer_results = self.layered_model(tf.convert_to_tensor(image), training=False)
        return {
            "predictions": layer_results[0][0],
            "features": layer_results[1][0],
        }
