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
        tf.config.set_visible_devices([], "GPU")
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

    # given an image object (usually coming from prepare_image_for_inference), an optional
    # filter taxon, and an optional cache_key, calculate vision results for the image, within the
    # filter taxon if provided, and save to a file if provided a cache_key
    def process_image(self, image, filter_taxon_id=None, cache_key=None):
        cache_path = self.cache_path_for_request(cache_key)
        cached_results = self.cached_results(cache_path)
        # vision results for this cache_key have already been generated, so use them
        if cached_results is not None:
            preds = cached_results
        else:
            preds = self.vision_model.predict(image)[0]
            self.cache_results(cache_path, preds)
        filtered_results = {}
        filter_taxon = None if filter_taxon_id is None else self.taxonomy.taxa[filter_taxon_id]
        sum_scores = 0
        # loop through vision results from highest score to lowest
        for arg in preds.argsort()[::-1]:
            score = preds[arg]
            # if there are already enough results, or scores are getting too low, then stop
            # NOTE: This may not be helpful and needs testing for accuracy and processing time
            if self.results_fully_populated(filtered_results, score):
                break
            taxon_id = self.taxonomy.leaf_class_to_taxon[arg]
            # ignore this result if the taxon is outside the requested filter taxon
            if filter_taxon is not None:
                result_taxon = self.taxonomy.taxa[taxon_id]
                if not result_taxon.is_or_descendant_of(filter_taxon):
                    continue
            filtered_results[taxon_id] = score
            sum_scores += score
        # normalize filtered scores to add to 1
        # NOTE: If there was a filter_taxon, or only a partial results list is returned (which is
        # the case now while using results_fully_populated), then this modifies the raw vision
        # scores. This ensures the scores of this method always sum to 1, even if the results have
        # been filtered
        for taxon_id in filtered_results:
            filtered_results[taxon_id] = filtered_results[taxon_id] / sum_scores
        return filtered_results
