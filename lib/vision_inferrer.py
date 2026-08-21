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
        self.embedding_model = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=full_model.layers[2].output,
        )

        self.infer = tf.function(
            lambda image: self.layered_model(image, training=False),
            input_signature=[
                tf.TensorSpec((1, 299, 299, 3), tf.float32)
            ],
            autograph=False,
        )
        self.infer_embedding = tf.function(
            lambda image: self.embedding_model(image, training=False),
            input_signature=[
                tf.TensorSpec((1, 299, 299, 3), tf.float32)
            ],
            autograph=False,
        )
        self.infer_embeddings = tf.function(
            lambda images: self.embedding_model(images, training=False),
            input_signature=[
                tf.TensorSpec((None, 299, 299, 3), tf.float32),
            ],
            autograph=False,
        )
        self.infer.get_concrete_function()

    # given an image object (usually coming from prepare_image_for_inference),
    # calculate vision results for the image
    def process_image(self, image):
        layer_results = self.infer(image)
        return {
            "predictions": layer_results[0][0],
            "features": layer_results[1][0],
        }

    def process_embedding(self, image):
        return self.infer_embedding(image)[0]

    def process_embeddings(self, images):
        return self.infer_embeddings(images)
