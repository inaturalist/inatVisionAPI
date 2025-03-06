from sys import platform

from lib.vision_inferrer import VisionInferrer
from lib.vision_inferrer_coreml import VisionInferrerCoreML
from lib.vision_inferrer_tflite import VisionInferrerTFLite
from lib.vision_inferrer_tf import VisionInferrerTF


class VisionInferrerFactory:
    @staticmethod
    def create(model_path: str) -> VisionInferrer:
        if "mlmodel" in model_path:
            assert platform == "darwin", "CoreML models can only be used on macOS"
            return VisionInferrerCoreML(model_path)
        elif "tflite" in model_path:
            return VisionInferrerTFLite(model_path)
        elif "h5" in model_path:
            return VisionInferrerTF(model_path)
        else:
            raise ValueError(f"Unsupported model format in path: {model_path}")
