import coremltools as ct
import numpy as np

from lib.geo_inferrer import GeoInferrer


class CoremlGeoPriorModelElev(GeoInferrer):

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.gpmodel = ct.models.MLModel(self.model_path)

    def predict(
        self, latitude: float, longitude: float, elevation: float
    ) -> np.ndarray:
        encoded_loc = GeoInferrer.encode_loc(
            [latitude], [longitude], [elevation]
        ).numpy()
        out_dict = self.gpmodel.predict({"input_1": encoded_loc})
        preds = out_dict["Identity"][0]
        return preds
