import coremltools as ct
from PIL import Image
import tensorflow as tf

from lib.vision_inferrer import VisionInferrer, VisionResults


class VisionInferrerCoreML(VisionInferrer):
    """Vision Inferrer for the CoreML variant of iNat vision models.
    Our implementation expects a single PIL image in the range [0, 255).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.prepare_model()

    def prepare_model(self):
        """initialize the CoreML model given the configured path"""
        self.model = ct.models.MLModel(self.model_path)
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name

    def process_image(self, image_tensor: tf.Tensor) -> VisionResults:
        """given an image object (coming from prepare_image_for_inference),
        calculate & return vision results for the image."""
        # coreml expects a PIL image so we have to convert from tf
        # first we convert from floats [0, 1) to ints [0, 255)
        image = tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)

        # Remove batch dimension if present and convert to NumPy array
        image_numpy = image.numpy()
        if image_numpy.ndim == 4:
            image_numpy = image_numpy[0]

        # Create PIL Image from NumPy array
        image_pil = Image.fromarray(image_numpy)

        out_dict = self.model.predict({self.input_name: image_pil})
        preds = out_dict["Identity"][0]

        # don't return features, not relevant for coreml at this point
        return {"predictions": preds, "features": None}
