import tensorflow as tf
import os
import hashlib
import pickle


class VisionInferrer:

    def __init__(self, model_path, taxonomy):
        self.model_path = model_path
        self.taxonomy = taxonomy
        self.prepare_tf_model()

    # initialize the TF model given the configured path
    def prepare_tf_model(self):
        # disable GPU processing
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

        self.vision_model = tf.keras.models.load_model(self.model_path, compile=False)

    # given a unique key, generate a path where vision results can be cached
    def cache_path_for_request(self, cache_key):
        if cache_key:
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            return os.path.join("./lib", "vision_cache", cache_hash)

    # given a path, return vision results cached at that path
    def cached_results(self, cache_path):
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as handle:
                results = pickle.loads(handle.read())
            return results

    # given a path, cache vision results in a file at that path
    def cache_results(self, cache_path, results):
        if cache_path is not None:
            with open(cache_path, "wb+") as cache_file:
                pickle.dump(results, cache_file)

    # only return results for up to 500 taxa, or until the scores are very low, whichever
    # comes first
    # NOTE: This may not be helpful and needs testing for accuracy and processing time
    def results_fully_populated(self, results, score):
        number_of_results = len(results)
        return (number_of_results >= 500 and score < 0.00000001) or number_of_results >= 5000

    # given an image object (usually coming from prepare_image_for_inference),
    # calculate vision results for the image
    def process_image(self, image):
        return self.vision_model.predict(image, verbose=0)[0]
