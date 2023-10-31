import csv
import os
import urllib
import hashlib
import magic
import time
import json
import tensorflow as tf
from statistics import mean
from PIL import Image
from lib.test_observation import TestObservation
from lib.inat_inferrer import InatInferrer


class VisionTesting:

    def __init__(self, config, **args):
        self.cmd_args = args
        self.inferrers = {}
        self.scores = {}
        score_types = ["matching_indices", "top1_distance_scores",
                       "top5_distance_scores", "top10_distance_scores",
                       "sum_ancestor_distance_scores", "average_ancestor_distance_scores"]
        for score_type in score_types:
            self.scores[score_type] = {
                "vision": {},
                "combined": {}
            }
        print("Models:")
        for index, model_config in enumerate(config["models"]):
            print(json.dumps(model_config, indent=4))
            model_name = model_config["name"] if "name" in model_config else f'Model {index}'
            model_config["name"] = model_name
            for score_type in score_types:
                self.scores[score_type]["vision"][index] = []
                self.scores[score_type]["combined"][index] = []

            self.inferrers[index] = InatInferrer(model_config)
        print("\n")
        self.upload_folder = "static/"

    def run(self):
        count = 0
        limit = self.cmd_args["limit"] or 100
        target_observation_id = self.cmd_args["observation_id"]
        start_time = time.time()
        try:
            with open(self.cmd_args["path"], "r") as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=",")
                for row in csv_reader:
                    observation = TestObservation(row)
                    if target_observation_id:
                        # if only one target observation was requested, test this row if it
                        # matches the request, otherwise skip it
                        if int(observation.observation_id) == target_observation_id:
                            inferrer_results = self.test_observation(observation)
                        else:
                            continue
                    else:
                        inferrer_results = self.test_observation(observation)
                    if inferrer_results is False:
                        # there was some problem processing this test observation. Continue but
                        # don't increment the counter so the requested number of observations
                        # will still be tested
                        continue
                    count += 1
                    self.append_to_aggregate_results(observation, inferrer_results)
                    if count % 10 == 0:
                        total_time = round(time.time() - start_time, 3)
                        remaining_time = round((limit - count) / (count / total_time), 3)
                        print(f'Processed {count} in {total_time} sec\testimated {remaining_time} sec remaining')
                    if count >= limit:
                        return
        except IOError as e:
            print(e)
            print("Testing run failed")

    # given an x, return the number of scores less than x. Otherwise return the number
    # of scores that are empty or greather than or equal to 100 (essentially the fails)
    def top_x(self, x, scores):
        if x is None:
            return len(list(filter(lambda score: score is None or score >= 100, scores)))
        return len(list(filter(lambda score: score is not None and score < x, scores)))

    # same as top_x, but returns the percentage of matching scores instead of the raw count
    def top_x_percent(self, x, scores):
        count = len(scores)
        top_x = self.top_x(x, scores)
        return round((top_x / count) * 100, 2)

    def print_scores(self):
        for index, inferrer in self.inferrers.items():
            all_metrics = {}
            for method in ["vision", "combined"]:
                scores = self.scores["matching_indices"][method][index]
                top1_distance_scores = self.scores["top1_distance_scores"][method][index]
                top5_distance_scores = self.scores["top5_distance_scores"][method][index]
                top10_distance_scores = self.scores["top10_distance_scores"][method][index]
                metrics = {}
                metrics["count"] = len(scores)
                metrics["top1"] = self.top_x(1, scores)
                metrics["top5"] = self.top_x(5, scores)
                metrics["top10"] = self.top_x(10, scores)
                metrics["notIn"] = self.top_x(None, scores)
                metrics["top1%"] = self.top_x_percent(1, scores)
                metrics["top5%"] = self.top_x_percent(5, scores)
                metrics["top10%"] = self.top_x_percent(10, scores)
                metrics["notIn%"] = self.top_x_percent(None, scores)
                metrics["top1∆"] = round(
                    (sum(top1_distance_scores) / metrics["count"]) * 100, 2)
                metrics["top5∆"] = round(
                    (sum(top5_distance_scores) / metrics["count"]) * 100, 2)
                metrics["top10∆"] = round(
                    (sum(top10_distance_scores) / metrics["count"]) * 100, 2)
                metrics["avg∆"] = round(
                    (mean(self.scores["average_ancestor_distance_scores"][method][index]) / metrics["count"]) * 100, 2)
                metrics["sum∆"] = round(
                    (mean(self.scores["sum_ancestor_distance_scores"][method][index]) / metrics["count"]) * 100, 2)
                all_metrics[method] = metrics

            print("method  " + "\t" + "\t".join(all_metrics["vision"].keys()))
            for method in ["vision", "combined"]:
                stat_label = inferrer.config["name"] + "-" + self.cmd_args["label"] + "-" + method
                print(f"{stat_label.ljust(10)}\t" + "\t".join(
                    str(value) for value in all_metrics[method].values()))
            print("\n")

    # NOTE: this is assuming no conversion is needed.
    # Ideally we'd reuse the inat_inferrer prepare_image_for_inference
    def prepare_image_for_inference(self, cache_path):
        image = tf.io.read_file(cache_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.central_crop(image, 0.875)
        image = tf.image.resize(image, [299, 299], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.expand_dims(image, 0)

    def assess_top_results(self, observation, top_results):
        match_index = None
        distance_scores = []
        for index, row in top_results.reset_index(drop=True).iterrows():
            if row["taxon_id"] == int(observation.taxon_id):
                match_index = index

            if index < 10:
                if row["taxon_id"] == int(observation.taxon_id):
                    # the taxa match, so the taxon distance score is 1
                    distance_scores.append(1)
                    break

                # if this is a top 10 result but not a match, append to taxon_scores
                # some measure of how far away this taxon is from the expected correct taxon using
                # (1 - [index of match in reversed target ancestry]/[lenth of target ancestry])
                # e.g. if the ancestry is 1/2/3/4/5/6/7/8 and this result has an ancestry of
                # 1/2/3/4/5, the match occcurs at taxon 5, which is in (reverse 0-indexed)
                # position 3 in the target taxon's ancestry, out of 8 taxa in that ancestry.
                # So the taxon score will be (1 - (3/8))^2, or (.625)^2, or 0.3090625
                # NOTE: This is experimental and needs testing
                try:
                    taxon_match_index = observation.taxon_ancestry[::-1].index(row["taxon_id"])
                except ValueError:
                    taxon_match_index = None
                if taxon_match_index:
                    distance_score = (1 - (taxon_match_index / len(observation.taxon_ancestry)))**2
                    distance_scores.append(distance_score)
                    break
                else:
                    distance_scores.append(0)
        return match_index, distance_scores

    def test_observation(self, observation):
        cache_path = self.download_photo(observation.photo_url)
        if cache_path is None or not os.path.exists(cache_path):
            return False
        if observation.lat == '' or observation.lng == '':
            return False

        iconic_taxon_id = None
        if observation.iconic_taxon_id != "" and self.cmd_args["filter_iconic"] is not False:
            iconic_taxon_id = int(observation.iconic_taxon_id)


        inferrer_scores = {}
        for index, inferrer in self.inferrers.items():
            lat = None
            lng = None
            filter_taxon = inferrer.lookup_taxon(iconic_taxon_id)
            if inferrer.geo_elevation_model and self.cmd_args["geo"]:
                lat = observation.lat
                lng = observation.lng
            try:
                inferrer_scores[index] = inferrer.predictions_for_image(
                    cache_path, lat, lng, filter_taxon
                )
            except Exception as e:
                print(e)
                print(f'\nError scoring observation {observation.observation_id}')
                return False
        return inferrer_scores

    def ancestor_distance_scores(self, observation, inferrer, results):
        reversed_target_ancestors = observation.taxon_ancestry[::-1]
        ancestor_distance_scores = []
        # for each top result
        for index, row in results.iterrows():
            result_ancestors = inferrer.taxonomy.df.query(
                f'left <= {row["left"]} and right >= {row["right"]}'
            ).sort_values("left", ascending=False).reset_index(drop=True)
            result_ancestor_match_index = None
            # find the most specific taxon in the result's taxon's ancestry that is also in
            # the target taxon's ancestry
            for ancestor_index, ancestor_row in result_ancestors.iterrows():
                if ancestor_row["taxon_id"] in reversed_target_ancestors:
                    result_ancestor_match_index = ancestor_index
                    break
            if result_ancestor_match_index is None:
                result_ancestor_match_index = len(reversed_target_ancestors)
            # calculate a score of how far from species the result matched the target
            ancestor_distance_scores.append((1 - (result_ancestor_match_index / len(reversed_target_ancestors)))**2)
        return ancestor_distance_scores

    def append_to_aggregate_results(self, observation, inferrer_scores):
        vision_indices = set()
        combined_indices = set()
        for index, results in inferrer_scores.items():
            # only look at the top 100 results for this testing

            top100_vision = results.sort_values("vision_score", ascending=False).head(100)
            top100_combined = results.sort_values("combined_score", ascending=False).head(100)

            vision_index, vision_taxon_distance_scores = self.assess_top_results(
                observation, top100_vision)
            combined_index, combined_taxon_distance_scores = self.assess_top_results(
                observation, top100_combined)

            vision_ancestor_distance_scores = self.ancestor_distance_scores(
                observation, self.inferrers[index], top100_vision.head(10))
            combined_ancestor_distance_scores = self.ancestor_distance_scores(
                observation, self.inferrers[index], top100_combined.head(10))

            self.scores["sum_ancestor_distance_scores"]["vision"][index].append(
                sum(vision_ancestor_distance_scores))
            self.scores["average_ancestor_distance_scores"]["vision"][index].append(
                mean(vision_ancestor_distance_scores))
            self.scores["sum_ancestor_distance_scores"]["combined"][index].append(
                sum(combined_ancestor_distance_scores))
            self.scores["average_ancestor_distance_scores"]["combined"][index].append(
                mean(combined_ancestor_distance_scores))

            vision_indices.add(vision_index)
            combined_indices.add(combined_index)
            self.scores["matching_indices"]["vision"][index].append(vision_index)
            self.scores["matching_indices"]["combined"][index].append(combined_index)
            # top1 distance score is just the taxon_distance_score if the first result
            self.scores["top1_distance_scores"]["vision"][index].append(
                vision_taxon_distance_scores[0])
            if len(combined_taxon_distance_scores) > 0:
                self.scores["top1_distance_scores"]["combined"][index].append(
                    combined_taxon_distance_scores[0])
            # for taxon_distance, top n is the max score of the top n results, or the
            # taxon_distance_score of the most closely related taxon in the first n results
            self.scores["top5_distance_scores"]["vision"][index].append(
                max(vision_taxon_distance_scores[0:5]))
            if len(combined_taxon_distance_scores) > 0:
                self.scores["top5_distance_scores"]["combined"][index].append(
                    max(combined_taxon_distance_scores[0:5]))
                self.scores["top10_distance_scores"]["vision"][index].append(
                    max(vision_taxon_distance_scores[0:10]))
                self.scores["top10_distance_scores"]["combined"][index].append(
                    max(combined_taxon_distance_scores[0:10]))

        # if len(combined_indices) > 1:
        #     print(f'Results of Observation: {observation.observation_id}: {combined_indices}')

    def download_photo(self, photo_url):
        checksum = hashlib.md5(photo_url.encode()).hexdigest()
        cache_path = os.path.join(self.upload_folder, "obs-" + checksum) + ".jpg"
        if os.path.exists(cache_path):
            return cache_path
        urllib.request.urlretrieve(photo_url, cache_path)
        mime_type = magic.from_file(cache_path, mime=True)
        if mime_type != "image/jpeg":
            im = Image.open(cache_path)
            rgb_im = im.convert("RGB")
            rgb_im.save(cache_path)
        return cache_path

    def debug(self, message):
        if self.cmd_args["debug"]:
            print(message)
