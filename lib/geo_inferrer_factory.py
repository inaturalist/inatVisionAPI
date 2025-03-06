from sys import platform

from lib.geo_inferrer import GeoInferrer
from lib.geo_inferrer_coreml import CoremlGeoPriorModelElev
from lib.geo_inferrer_tflite import TFLiteGeoPriorModelElev
from lib.geo_inferrer_tf import TFGeoPriorModelElev


class GeoInferrerFactory:
    @staticmethod
    def create(model_path: str) -> GeoInferrer:
        if "mlmodel" in model_path:
            assert platform == "darwin", "CoreML models can only be used on macOS"
            return CoremlGeoPriorModelElev(model_path)
        elif "tflite" in model_path:
            return TFLiteGeoPriorModelElev(model_path)
        elif "h5" in model_path:
            return TFGeoPriorModelElev(model_path)
        else:
            raise ValueError(f"Unsupported model format in path: {model_path}")
