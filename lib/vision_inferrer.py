import tensorflow as tf


class VisionInferrer:

    def __init__(self, model_path, signature_layer_name=None):
        self.model_path = model_path
        self.signature_layer_name = signature_layer_name
        self.prepare_tf_model()

    # initialize the TF model given the configured path
    def prepare_tf_model(self):
        # disable GPU processing
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"

        self.vision_model = tf.keras.models.load_model(self.model_path, compile=False)
        if self.signature_layer_name is not None:
            self.signature_model = tf.keras.Model(
                inputs=self.vision_model.inputs,
                outputs=self.vision_model.get_layer(self.signature_layer_name).output
            )
            self.signature_model.compile()

    # given an image object (usually coming from prepare_image_for_inference),
    # calculate vision results for the image
    def process_image(self, image):
        return self.vision_model(tf.convert_to_tensor(image), training=False)[0]

    def signature_for_image(self, image):
        if not hasattr(self, "signature_model"):
            return
        return self.signature_model(tf.convert_to_tensor(image), training=False)[0].numpy()
