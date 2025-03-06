from abc import ABC, abstractmethod
from typing import Optional, TypedDict

import numpy as np
import tensorflow as tf


class VisionResults(TypedDict):
    predictions: np.ndarray
    features: Optional[np.ndarray]


class VisionInferrer(ABC):
    @abstractmethod
    def __init__(self, model_path: str):
        """Subclasses must implement this constructor."""
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Initialize the model.

        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def process_image(self, image: tf.Tensor) -> VisionResults:
        """
        given an image object (usually coming from prepare_image_for_inference),
        calculate vision results for the image

        Subclasses must implement this method.
        """
        pass
