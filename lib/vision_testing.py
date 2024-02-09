import os
import hashlib
import magic
import time
import json
import pandas as pd
import asyncio
import aiohttp
import aiofiles
import aiofiles.os
import re
import traceback
from datetime import datetime
from PIL import Image
from lib.test_observation import TestObservation
from lib.inat_inferrer import InatInferrer


class VisionTesting:

    def __init__(self, config, **args):
        self.cmd_args = args
        self.inferrers = {}
        self.upload_folder = "static/"
        currentDatetime = datetime.now()
        self.start_timestamp = currentDatetime.strftime("%Y%m%d")
        self.set_run_hash(config)

        print("Models:")
        for inferrer_index, model_config in enumerate(config["models"]):
            print(json.dumps(model_config, indent=4))
            model_name = model_config["name"] if "name" in model_config \
                else f"Model {inferrer_index}"
            model_config["name"] = model_name
            self.inferrers[inferrer_index] = InatInferrer(model_config)

    def set_run_hash(self, config):
        run_hash_dict = dict(config)
        run_hash_dict.update(self.cmd_args)
        dhash = hashlib.md5()
        encoded = json.dumps(run_hash_dict, sort_keys=True).encode()
        dhash.update(encoded)
        self.run_hash = dhash.hexdigest()

    def export_path(self, filename_addition=None, label=None):
        if label is None:
            label = self.cmd_args["label"]
        export_path = f"test-results-{self.start_timestamp}-{label}-{self.run_hash}"
        if filename_addition:
            export_path += f"-{filename_addition}"
        export_path += ".csv"
        if self.cmd_args["data_dir"]:
            export_path = os.path.join(self.cmd_args["data_dir"], export_path)
        return export_path

    async def run_async(self):
        if self.cmd_args["data_dir"]:
            for file in sorted(os.listdir(self.cmd_args["data_dir"])):
                exported_data_filename_match = re.search(r"test-obs-[0-9]{8}-(.*).csv", file)
                if exported_data_filename_match is None:
                    continue
                label = exported_data_filename_match.group(1)
                path = os.path.join(self.cmd_args["data_dir"], file)
                print(f"\nProcessing {file}")
                await self.testObservationsAtPath(path, label)
                self.display_and_save_results(label)
        else:
            print(f"\nProcessing {self.cmd_args['path']}")
            await self.testObservationsAtPath(self.cmd_args["path"], self.cmd_args["label"])
            self.display_and_save_results(self.cmd_args["label"])

    async def testObservationsAtPath(self, path, label):
        N_WORKERS = 5
        self.limit = self.cmd_args["limit"] or 100
        target_observation_id = self.cmd_args["observation_id"]
        self.start_time = time.time()
        self.queued_counter = 0
        self.processed_counter = 0
        self.test_observations = {}

        async with aiohttp.ClientSession() as self.session:
            self.queue = asyncio.Queue()
            self.workers = [
                asyncio.create_task(self.worker_task()) for _ in range(N_WORKERS)
            ]
            df = pd.read_csv(
                path,
                usecols=[
                    "observation_id",
                    "observed_on",
                    "iconic_taxon_id",
                    "taxon_id",
                    "taxon_ancestry",
                    "lat",
                    "lng",
                    "photo_url"
                ],
                dtype={
                    "iconic_taxon_id": float,
                    "taxon_id": int,
                    "lat": float,
                    "lng": float
                }
            )
            for index, observation in df.iterrows():
                if target_observation_id and observation.observation_id != target_observation_id:
                    continue
                obs = TestObservation(observation.to_dict())
                self.test_observations[obs.observation_id] = obs
                self.queue.put_nowait(obs.observation_id)

            # processes the queue
            await self.queue.join()
            # stop the workers
            for worker in self.workers:
                worker.cancel()

    async def worker_task(self):
        while not self.queue.empty():
            observation_id = await self.queue.get()
            try:
                if self.processed_counter >= self.limit:
                    continue
                observation = self.test_observations[observation_id]
                await self.test_observation_async(observation)
                if observation.inferrer_scores is None:
                    continue
                self.processed_counter += 1
                self.report_progress()

            except Exception as err:
                print(f"\nObservation: {observation_id} failed")
                print(traceback.format_exc())
                print(err)

            finally:
                self.queue.task_done()

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

    def display_and_save_results(self, label):
        scored_observations = list(filter(
            lambda observation: len(observation.summarized_results) > 0,
            self.test_observations.values()
        ))
        if len(scored_observations) == 0:
            return

        # extract a summary score for each observation for each inferrer and scoring method
        keys = scored_observations[0].summarized_results[0].keys()
        all_obs_scores = []
        for obs in scored_observations:
            for inferrer_index, inferrer in self.inferrers.items():
                for method in keys:
                    scores = {
                        "inferrer_name": inferrer.config["name"],
                        "label": label,
                        "method": method,
                        "uuid": obs.observation_id
                    }
                    scores.update(obs.summarized_results[inferrer_index][method])
                    all_obs_scores.append(scores)
        pd.set_option("display.max_rows", None)
        all_obs_scores_df = pd.DataFrame(all_obs_scores)
        all_obs_scores_df["run_label"] = all_obs_scores_df[
            ["inferrer_name", "label", "method"]
        ].agg("-".join, axis=1)
        observations_export_path = self.export_path("observations", label=label)
        all_obs_scores_df.to_csv(observations_export_path)

        # generate a summary score of all observations for each inferrer and scoring method
        grouped_stats = all_obs_scores_df.groupby("run_label").agg(
            inferrer_name=("inferrer_name", "max"),
            label=("label", "max"),
            method=("method", "max"),
            count=("label", "count"),
        )
        top1 = all_obs_scores_df.query("matching_index == 0").groupby("run_label").agg(
            top1=("label", "count"),
        )
        top5 = all_obs_scores_df.query("matching_index < 5").groupby("run_label").agg(
            top5=("label", "count"),
        )
        top10 = all_obs_scores_df.query("matching_index < 10").groupby("run_label").agg(
            top10=("label", "count"),
        )
        notIn = all_obs_scores_df.query("matching_index.isna()").groupby("run_label").agg(
            notIn=("label", "count"),
        )
        grouped_stats = grouped_stats.merge(
            top1, how="left", left_on="run_label", right_on="run_label"
        )
        grouped_stats = grouped_stats.merge(
            top5, how="left", left_on="run_label", right_on="run_label"
        )
        grouped_stats = grouped_stats.merge(
            top10, how="left", left_on="run_label", right_on="run_label"
        )
        grouped_stats = grouped_stats.merge(
            notIn, how="left", left_on="run_label", right_on="run_label"
        )
        grouped_stats["top1%"] = round((grouped_stats["top1"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["top5%"] = round((grouped_stats["top5"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["top10%"] = round((grouped_stats["top10"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["notIn%"] = round((grouped_stats["notIn"] / grouped_stats["count"]) * 100, 2)

        agg_stats = all_obs_scores_df.groupby("run_label").agg(
            average_results_count=("results_count", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean")
        )
        grouped_stats = grouped_stats.merge(
            agg_stats, how="left", left_on="run_label", right_on="run_label"
        )
        grouped_stats["average_results_count"] = grouped_stats["average_results_count"].round(4)
        grouped_stats["precision"] = grouped_stats["precision"].round(4)
        grouped_stats["recall"] = grouped_stats["recall"].round(4)
        grouped_stats["f1"] = grouped_stats["f1"].round(4)
        grouped_stats = grouped_stats.sort_index(ascending=False)
        print(grouped_stats[grouped_stats.columns.difference(["inferrer_name", "label", "method"])])

        grouped_stats_export_path = self.export_path("summary", label=label)
        grouped_stats.to_csv(grouped_stats_export_path)

    async def test_observation_async(self, observation):
        cache_path = await self.download_photo_async(observation.photo_url)

        # due to asynchronous processing, the requested limit of observations to test
        # has been reached, so do not test this observation. The rest of this method
        # will be processed synchronously, so no need to check this again this method
        if self.processed_counter >= self.limit:
            return

        if cache_path is None \
           or not os.path.exists(cache_path) \
           or observation.lat == "" \
           or observation.lng == "":
            return

        iconic_taxon_id = None
        if observation.iconic_taxon_id != "" and self.cmd_args["filter_iconic"] is not False:
            iconic_taxon_id = observation.iconic_taxon_id

        inferrer_scores = {}
        for inferrer_index, inferrer in self.inferrers.items():
            lat = None
            lng = None
            filter_taxon = inferrer.lookup_taxon(iconic_taxon_id)
            if inferrer.geo_elevation_model and self.cmd_args["geo"]:
                lat = observation.lat
                lng = observation.lng
            try:
                inferrer_all_scores = inferrer.predictions_for_image(
                    cache_path, lat, lng, filter_taxon
                )
                # only look at the top 100 results for this testing
                inferrer_scores[inferrer_index] = {
                    "vision": inferrer_all_scores.sort_values(
                        "vision_score", ascending=False
                    ).reset_index(drop=True).head(100),
                    "combined": inferrer_all_scores.sort_values(
                        "combined_score", ascending=False
                    ).reset_index(drop=True).head(100),
                    "combined_nearby": inferrer_all_scores.query(
                        "geo_score >= geo_threshold"
                    ).sort_values(
                        "combined_score", ascending=False
                    ).reset_index(drop=True).head(100),
                }
            except Exception as e:
                print(f"Error scoring observation {observation.observation_id}")
                print(e)
                return

        observation.inferrer_scores = inferrer_scores
        self.summarize_results(observation)

    def matching_index(self, observation, results):
        matching_indices = results.index[
            results["taxon_id"] == observation.taxon_id
        ].tolist()
        return matching_indices[0] if len(matching_indices) > 0 else None

    def summarize_results(self, observation):
        for inferrer_index, results in observation.inferrer_scores.items():
            observation.summarized_results[inferrer_index] = {}
            for results_index, results_scores in results.items():
                self.summarize_result_subset(observation, inferrer_index, results, results_index)
                self.summarize_result_subset(
                    observation, inferrer_index, results, results_index, cutoff=True
                )

    def summarize_result_subset(self, observation, index, results, subset, cutoff=False):
        working_results = results[subset]
        summary_label = subset
        if cutoff:
            summary_label += "-cutoff"
            score_column = "vision_score" if subset == "vision" else "combined_score"
            values = results[subset].head(1)[score_column].values
            if len(values) == 0:
                top_score = 0
            else:
                top_score = values[0]
            working_results = results[subset].query(
                f"{score_column} > {top_score * 0.001}"
            ).head(10)

        matching_index = self.matching_index(observation, working_results)

        results_count = len(working_results.index)
        summary = {
            "results_count": results_count,
            "matching_index": matching_index,
            "recall": 1 if matching_index is not None else 0,
            "precision": 0 if matching_index is None else 1 / results_count,
        }
        sum_of_precision_and_recall = summary["precision"] + summary["recall"]
        summary["f1"] = 0 if sum_of_precision_and_recall == 0 else (
            2 * summary["precision"]
        ) / sum_of_precision_and_recall
        observation.summarized_results[index][summary_label] = summary

    async def download_photo_async(self, photo_url):
        checksum = hashlib.md5(photo_url.encode()).hexdigest()
        cache_path = os.path.join(self.upload_folder, "obs-" + checksum) + ".jpg"
        if await aiofiles.os.path.exists(cache_path):
            return cache_path
        async with self.session.get(photo_url) as resp:
            if resp.status == 200:
                f = await aiofiles.open(cache_path, mode="wb")
                await f.write(await resp.read())
                await f.close()
        if not os.path.exists(cache_path):
            return
        mime_type = magic.from_file(cache_path, mime=True)
        if mime_type != "image/jpeg":
            im = Image.open(cache_path)
            rgb_im = im.convert("RGB")
            rgb_im.save(cache_path)
        return cache_path

    def debug(self, message):
        if self.cmd_args["debug"]:
            print(message)

    def report_progress(self):
        if self.processed_counter % 10 == 0:
            total_time = round(time.time() - self.start_time, 2)
            remaining_time = round((
                self.limit - self.processed_counter
            ) / (self.processed_counter / total_time), 2)
            rate = round(self.processed_counter / total_time, 2)
            print(
                f"Processed {self.processed_counter} in {total_time} sec  \t"
                f"{rate}/sec  \t"
                f"estimated {remaining_time} sec remaining\t"
            )

    # def assess_top_results(self, observation, top_results):
    #     match_index = None
    #     distance_scores = []
    #     for index, row in top_results.reset_index(drop=True).iterrows():
    #         if row["taxon_id"] == observation.taxon_id:
    #             match_index = index

    #         if index < 10:
    #             if row["taxon_id"] == observation.taxon_id:
    #                 # the taxa match, so the taxon distance score is 1
    #                 distance_scores.append(1)
    #                 break

    #             # if this is a top 10 result but not a match, append to taxon_scores
    #             # some measure of how far away this taxon is from the expected correct taxon using
    #             # (1 - [index of match in reversed target ancestry]/[lenth of target ancestry])
    #             # e.g. if the ancestry is 1/2/3/4/5/6/7/8 and this result has an ancestry of
    #             # 1/2/3/4/5, the match occcurs at taxon 5, which is in (reverse 0-indexed)
    #             # position 3 in the target taxon's ancestry, out of 8 taxa in that ancestry.
    #             # So the taxon score will be (1 - (3/8))^2, or (.625)^2, or 0.3090625
    #             # NOTE: This is experimental and needs testing
    #             try:
    #                 taxon_match_index = observation.taxon_ancestry[::-1].index(row["taxon_id"])
    #             except ValueError:
    #                 taxon_match_index = None
    #             if taxon_match_index:
    #                 distance_score = (1 - (taxon_match_index / len(observation.taxon_ancestry)))**2
    #                 distance_scores.append(distance_score)
    #                 break
    #             else:
    #                 distance_scores.append(0)
    #     return match_index, distance_scores
