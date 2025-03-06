import tensorflow as tf

from lib.vision_inferrer import VisionInferrer, VisionResults


class VisionInferrerTFLite(VisionInferrer):
    """Vision Inferrer for the tflite variant of iNat vision models.
    Our implementation expects inputs in the range [0, 255).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.prepare_model()

    def prepare_model(self):
        """initialize the tflite model given the configured path"""
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def process_image(self, image_tensor: tf.Tensor) -> VisionResults:
        """given an image object (coming from prepare_image_for_inference),
        calculate & return vision results for the image."""
        # tflite expects an image in range [0, 255] not [0, 1]
        image_tensor = image_tensor * 255

        # set the input to tflite model
        input_dtype = self.input_details[0]["dtype"]
        input_data = image_tensor.numpy().astype(input_dtype)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # execute the tflite model
        self.interpreter.invoke()

        # extract the output
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        # don't return features, not relevant for tflite at this point
        return {"predictions": output_data[0], "features": None}
