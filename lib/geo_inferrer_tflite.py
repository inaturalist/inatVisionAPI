import numpy as np
import tensorflow as tf

from lib.geo_inferrer import GeoInferrer


class TFLiteGeoPriorModelElev(GeoInferrer):

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def predict(
        self, latitude: float, longitude: float, elevation: float
    ) -> np.ndarray:
        encoded_loc = GeoInferrer.encode_loc(
            [latitude], [longitude], [elevation]
        ).numpy()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_dtype = input_details[0]["dtype"]
        encoded_loc = encoded_loc.astype(input_dtype)

        self.interpreter.set_tensor(
            input_details[0]["index"],
            encoded_loc,
        )
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]["index"])
        return output_data[0]
